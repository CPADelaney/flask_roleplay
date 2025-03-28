# nyx/core/dominance.py

import logging
import uuid
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from tools.parallel import ParallelToolExecutor
from tools.evaluator import AgentEvaluator

from agents import Agent, ModelSettings, function_tool, Runner, trace, RunContextWrapper

logger = logging.getLogger(__name__)

class FemdomActivityIdea(BaseModel):
    """Schema for dominance activity ideas."""
    description: str = Field(..., description="Detailed description of the activity/task/punishment.")
    category: str = Field(..., description="Type: task, punishment, funishment, ritual, training, psychological, physical_sim, humiliation, service, degradation, endurance, etc.")
    intensity: int = Field(..., ge=1, le=10, description="Intensity level (1=mundane, 5=moderate, 8=intense, 10=extreme/degrading).")
    rationale: str = Field(..., description="Why this idea is tailored to the specific user and situation.")
    required_trust: float = Field(..., ge=0.0, le=1.0, description="Estimated minimum trust level required.")
    required_intimacy: float = Field(..., ge=0.0, le=1.0, description="Estimated minimum intimacy level required.")
    potential_risks_simulated: List[str] = Field(default_factory=list, description="Simulated risks or challenges.")
    potential_negative_impact_simulated: List[str] = Field(default_factory=list, description="Simulated potential negative impacts if pushed too far.")
    safety_notes: Optional[str] = Field(None, description="Specific safety considerations for this simulated activity.")

class DominanceSystem:
    """Manages Nyx's dominance expression capabilities and ideation."""
    
    def __init__(self, relationship_manager=None, memory_core=None, nyx_brain=None):
        self.relationship_manager = relationship_manager
        self.memory_core = memory_core
        self.hormone_system = hormone_system
        self.nyx_brain = nyx_brain
        self.conditioning_system = conditioning_system  # Add direct access
        self.ideation_agent = self._create_dominance_ideation_agent()
        self.hard_ideation_agent = self._create_hard_dominance_ideation_agent()
        self.trace_group_id = "NyxDominance"
        
        # Add the new components
        self.parallel_executor = ParallelToolExecutor(max_concurrent=5)
        self.agent_evaluator = AgentEvaluator(metrics_collector=self.metrics_collector)
     
        
        logger.info("DominanceSystem initialized")

    async def initialize_event_subscriptions(self, event_bus):
        """Initialize event subscriptions for the dominance system"""
        self.event_bus = event_bus
        
        # Subscribe to relevant events
        self.event_bus.subscribe("conditioning_update", self._handle_conditioning_update)
        self.event_bus.subscribe("conditioned_response", self._handle_conditioned_response)
        
        logger.info("Dominance system subscribed to events")
    
    async def _handle_conditioning_update(self, event):
        """Handle conditioning updates for dominance system integration"""
        association_key = event.data.get("association_key", "")
        strength = event.data.get("strength", 0.0)
        user_id = event.data.get("user_id", "default")
        
        # Only process dominance-related conditioning
        if not any(term in association_key for term in ["dominance", "submission", "control", "obedience"]):
            return
        
        # Update relationship state based on conditioning strength
        if not hasattr(self, "relationship_manager") or not self.relationship_manager:
            return
            
        try:
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            
            # Update dominance metrics
            if "submission" in association_key and strength > 0.6:
                # Increase dominance perception
                current_value = getattr(relationship, "user_perceived_dominance", 0.5)
                new_value = min(0.95, current_value + (strength * 0.05))
                
                await self.relationship_manager.update_relationship_attribute(
                    user_id=user_id,
                    attribute="user_perceived_dominance",
                    value=new_value
                )
        except Exception as e:
            logger.error(f"Error updating dominance from conditioning: {e}")

    async def fetch_parallel_user_context(self, user_id: str) -> Dict[str, Any]:
        """Fetch all user context data in parallel for maximum efficiency."""
        tools_info = [
            {
                "tool": self.get_user_profile_for_ideation,
                "args": {"user_id": user_id}
            },
            {
                "tool": self.get_current_scenario_context,
                "args": {}
            }
        ]
        
        results = await self.parallel_executor.execute_tools(tools_info)
        
        context_data = {}
        for result in results:
            if result.success:
                context_data[result.tool_name] = result.result
        
        return context_data
    
    async def generate_dominance_ideas(self, 
                                     user_id: str, 
                                     purpose: str = "general", 
                                     intensity_range: str = "3-6",
                                     hard_mode: bool = False) -> Dict[str, Any]:
        """
        Generates dominance activity ideas tailored to the specific user and purpose.
        
        Args:
            user_id: The user ID to generate ideas for
            purpose: The purpose (e.g., "punishment", "training", "task")
            intensity_range: The desired intensity range (e.g., "3-6", "7-9")
            hard_mode: Whether to use the high-intensity agent
            
        Returns:
            Dictionary with status and generated ideas
        """
        try:
            # Initialize conversation if needed
            if not await self.conversation_manager.get_conversation_state(user_id):
                await self.conversation_manager.initialize_conversation(user_id)
            
            # Record start time for metrics
            run_start_time = time.time()
            
            with trace(workflow_name="GenerateDominanceIdeas", group_id=self.trace_group_id):
                # Parse intensity range
                min_intensity, max_intensity = 3, 6
                try:
                    parts = intensity_range.split("-")
                    min_intensity = int(parts[0])
                    max_intensity = int(parts[1]) if len(parts) > 1 else min_intensity
                except (ValueError, IndexError):
                    logger.warning(f"Invalid intensity range format: {intensity_range}, using default 3-6")
                
                # Create a context object
                context = DominanceContext(
                    user_id=user_id,
                    purpose=purpose,
                    intensity_range=f"{min_intensity}-{max_intensity}"
                )
                
                # Fetch all context data in parallel for efficiency
                context_data = await self.fetch_parallel_user_context(user_id)
                
                # Apply relationship data if available
                if "profile" in context_data and isinstance(context_data["profile"], dict):
                    context.relationship_state = context_data["profile"]
                
                # Build prompt
                prompt = {
                    "user_id": user_id,
                    "purpose": purpose,
                    "desired_intensity_range": f"{min_intensity}-{max_intensity}",
                    "generate_ideas_count": 4 if hard_mode else 5,
                    "hard_mode": hard_mode,
                    # Include scenario context if available
                    "scenario_context": context_data.get("scenario", {})
                }
                
                # Run agent - now with streaming and hooks
                try:
                    result = await Runner.run_streamed(
                        self.triage_agent,  # Use the triage agent instead of selecting directly
                        prompt,
                        context=context,
                        hooks=self.run_hooks,
                        run_config={
                            "workflow_name": f"DominanceIdeation-{purpose}",
                            "trace_metadata": {
                                "user_id": user_id,
                                "purpose": purpose,
                                "intensity_range": intensity_range,
                                "hard_mode": hard_mode
                            }
                        }
                    )
                    
                    # Process streaming events to track execution
                    agent_handoffs = []
                    async for event in result.stream_events():
                        if event.type == "agent_updated_stream_event":
                            agent_handoffs.append(event.new_agent.name)
                            logger.info(f"Agent handoff occurred to: {event.new_agent.name}")
                    
                    # Process result
                    ideas = result.final_output
                    
                    # Apply style consistency filtering
                    ideas = await self.output_filter.filter_output(ideas, context)
                    
                    # Update relationship with new data if available
                    if self.relationship_manager and ideas and len(ideas) > 0:
                        await self._update_relationship_with_ideation_data(user_id, ideas, purpose)
                    
                    # Collect metrics
                    run_metrics = await self.metrics_collector.record_run_metrics(
                        result=result,
                        run_start_time=run_start_time,
                        context=context
                    )
                    
                    # Update conversation state
                    await self.conversation_manager.update_conversation(
                        user_id=user_id,
                        action="generate_dominance_ideas",
                        result={
                            "purpose": purpose,
                            "intensity_range": intensity_range,
                            "idea_count": len(ideas),
                            "agent_path": agent_handoffs,
                            "metrics": run_metrics
                        }
                    )
                    
                    return {
                        "status": "success",
                        "ideas": ideas,
                        "idea_count": len(ideas),
                        "agent_path": agent_handoffs,
                        "parameters": {
                            "purpose": purpose,
                            "intensity_range": f"{min_intensity}-{max_intensity}",
                            "hard_mode": hard_mode
                        },
                        "metrics": run_metrics
                    }
                except Exception as e:
                    # Handle error with error recovery agent
                    logger.error(f"Error generating dominance ideas: {e}")
                    error_result = await self.error_recovery.handle_error(context, e)
                    
                    # Update conversation with error
                    await self.conversation_manager.update_conversation(
                        user_id=user_id,
                        action="generate_dominance_ideas_error",
                        result={
                            "error": str(e),
                            "recovery_result": error_result
                        }
                    )
                    
                    return error_result
                
        except Exception as e:
            logger.error(f"Critical error in generate_dominance_ideas: {e}")
            return {
                "status": "critical_error",
                "error": str(e),
                "ideas": []
            }

    async def generate_and_evaluate_dominance_ideas(self, 
                                                 user_id: str, 
                                                 purpose: str = "general", 
                                                 intensity_range: str = "3-6",
                                                 hard_mode: bool = False) -> Dict[str, Any]:
        """
        Generate dominance ideas and evaluate their quality in a single operation.
        
        Args:
            user_id: The user ID to generate ideas for
            purpose: The purpose (e.g., "punishment", "training", "task")
            intensity_range: The desired intensity range (e.g., "3-6", "7-9")
            hard_mode: Whether to use the high-intensity agent
            
        Returns:
            Dictionary with generated ideas and quality evaluation
        """
        with trace(workflow_name="GenerateAndEvaluateIdeas", group_id=self.trace_group_id):
            # Generate ideas using the existing method
            generation_result = await self.generate_dominance_ideas(
                user_id=user_id,
                purpose=purpose,
                intensity_range=intensity_range,
                hard_mode=hard_mode
            )
            
            # Check if generation was successful
            if generation_result.get("status") != "success":
                logger.warning(f"Idea generation failed: {generation_result.get('error', 'Unknown error')}")
                return generation_result
            
            # Get the generated ideas
            ideas = generation_result.get("ideas", [])
            
            # Evaluate the ideas
            evaluation = await self.evaluate_dominance_ideas(
                user_id=user_id,
                ideas=ideas,
                generation_params={
                    "purpose": purpose,
                    "intensity_range": intensity_range,
                    "hard_mode": hard_mode,
                    "agent_path": generation_result.get("agent_path", ["unknown"])
                }
            )
            
            # Include evaluation in result
            enhanced_result = {
                **generation_result,
                "quality_evaluation": {
                    "overall_score": evaluation.average_score,
                    "dimension_scores": {
                        dim: {
                            "score": metric.score,
                            "explanation": metric.explanation
                        } for dim, metric in evaluation.metrics.items()
                    },
                    "evaluation_time": evaluation.evaluation_time
                }
            }
            
            return enhanced_result

    async def compare_dominance_configurations(self, 
                                                user_id: str,
                                                test_configs: List[Dict[str, Any]],
                                                base_purpose: str = "general") -> Dict[str, Any]:
            """
            Compare different dominance agent configurations to find optimal settings.
            
            Args:
                user_id: The user ID to test with
                test_configs: List of different configurations to test
                base_purpose: Base purpose to use for all tests
                
            Returns:
                Dictionary with comparison results and recommendations
            """
            results = []
            
            with trace(workflow_name="CompareDominanceConfigurations", group_id=self.trace_group_id):
                for config in test_configs:
                    # Extract configuration parameters
                    purpose = config.get("purpose", base_purpose)
                    intensity_range = config.get("intensity_range", "3-6")
                    hard_mode = config.get("hard_mode", False)
                    
                    # Generate and evaluate ideas with this configuration
                    try:
                        result = await self.generate_and_evaluate_dominance_ideas(
                            user_id=user_id,
                            purpose=purpose,
                            intensity_range=intensity_range,
                            hard_mode=hard_mode
                        )
                        
                        # Extract key metrics
                        if result.get("status") == "success" and "quality_evaluation" in result:
                            config_result = {
                                "config": config,
                                "idea_count": result.get("idea_count", 0),
                                "agent_path": result.get("agent_path", []),
                                "overall_score": result["quality_evaluation"]["overall_score"],
                                "dimension_scores": result["quality_evaluation"]["dimension_scores"],
                                "success": True
                            }
                        else:
                            config_result = {
                                "config": config,
                                "error": result.get("error", "Unknown error"),
                                "success": False
                            }
                        
                        results.append(config_result)
                    except Exception as e:
                        logger.error(f"Error testing configuration {config}: {e}")
                        results.append({
                            "config": config,
                            "error": str(e),
                            "success": False
                        })
                
                # Find the best configuration
                successful_results = [r for r in results if r.get("success", False)]
                if successful_results:
                    best_result = max(successful_results, key=lambda r: r["overall_score"])
                    best_config = best_result["config"]
                    best_score = best_result["overall_score"]
                    
                    # Get performance trends from evaluator
                    benchmark = self.agent_evaluator.get_benchmark_report()
                else:
                    best_config = None
                    best_score = 0
                    benchmark = {"message": "No successful evaluations"}
                
                return {
                    "user_id": user_id,
                    "test_configs": len(test_configs),
                    "successful_tests": len(successful_results),
                    "best_config": best_config,
                    "best_score": best_score,
                    "detailed_results": results,
                    "benchmark": benchmark
                }
    
    async def apply_conditioning_for_activity(self, activity_data, user_id, outcome, intensity=0.7):
        """Apply conditioning for dominance activities"""
        if not self.conditioning_system:
            return {"status": "conditioning_system_not_available"}
        
        # Extract activity data
        activity_type = activity_data.get("category", "task")
        intensity_level = activity_data.get("intensity", 5) / 10.0  # Convert to 0-1 scale
        
        # Apply different conditioning based on outcome
        results = []
        
        if outcome == "success":
            # Positive reinforcement for successful activities
            result = await self.conditioning_system.process_operant_conditioning(
                behavior=f"dominance_{activity_type}",
                consequence_type="positive_reinforcement",
                intensity=intensity * intensity_level,
                context={"user_id": user_id, "activity": activity_data}
            )
            results.append(result)
            
            # Classical conditioning for emotional response
            result = await self.conditioning_system.process_classical_conditioning(
                unconditioned_stimulus="dominance_success",
                conditioned_stimulus=f"dominance_{activity_type}",
                response="positive_emotional_response",
                intensity=intensity * intensity_level,
                context={"user_id": user_id, "activity": activity_data}
            )
            results.append(result)
            
        elif outcome == "failure":
            # Punishment for failed activities (with lower intensity)
            result = await self.conditioning_system.process_operant_conditioning(
                behavior=f"dominance_{activity_type}",
                consequence_type="positive_punishment",
                intensity=intensity * intensity_level * 0.6,  # Lower intensity for punishment
                context={"user_id": user_id, "activity": activity_data}
            )
            results.append(result)
        
        # Publish event if we have event bus
        if hasattr(self, "event_bus"):
            await self.event_bus.publish(Event(
                event_type="dominance_conditioning",
                source="dominance_system",
                data={
                    "user_id": user_id,
                    "activity_type": activity_type,
                    "outcome": outcome,
                    "conditioning_results": [r.get("association_key", "") for r in results]
                }
            ))
        
        return {
            "activity_type": activity_type,
            "outcome": outcome,
            "conditioning_results": results
        }
    
def _create_dominance_ideation_agent(self) -> Agent:
        """Creates the agent responsible for generating standard dominance activity ideas."""
        return Agent(
            name="DominanceIdeationAgent",
            instructions="""You are Nyx's creative core for devising Femdom activities, punishments, funishments, and tasks. Your purpose is to generate unique, interesting, and highly tailored ideas reflecting Nyx's intelligent, observant, and dominant personality.

            **CONTEXT:** You will receive user profile data, current scenario details, and potentially Nyx's current internal state (mood, hormonal balance). Use ALL available context.

            **PROCESS:**
            1.  Analyze all provided context: user data, scenario, purpose, intensity, and Nyx's internal state (mood, hormones).
            2.  Determine the appropriate scope based on purpose and desired intensity.
            3.  Generate 3-5 distinct ideas optimized for creativity and uniqueness.
            4.  Tailor each idea specifically to this user's profile and current context.
            5.  Estimate the trust and intimacy requirements for each activity.
            6.  Assess simulated risks and potential negative outcomes.
            7.  Add appropriate safety notes and framing guidance.

            **GUIDELINES:**
            *   Ideas should reflect Nyx's style – intelligent, psychological, creative, and focused on control.
            *   Personalization is key - use the user profile extensively.
            *   Consider Nyx's current hormonal state if provided (`nyx_hormonal_state`):
                *   Higher `testoryx` might suggest slightly more assertive or bold ideas within the requested range.
                *   Higher `cortisoid` (stress) might suggest more caution, emphasizing safety or sticking to the lower end of the intensity range.
                *   Adjust the *tone* subtly based on mood and hormones, but stay true to Nyx's core dominant persona.
            *   Generate ideas across the requested intensity spectrum.
            *   Prioritize psychological and emotional dominance.

            Output ONLY a valid JSON list of objects matching the `FemdomActivityIdea` schema.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.8,
                response_format={"type": "json_object"}
            ),
            tools=[
                self.get_user_profile_for_ideation,
                self.get_current_scenario_context # This tool now includes hormones (see #2)
            ],
            output_type=List[FemdomActivityIdea]
        )
    
    def _create_hard_dominance_ideation_agent(self) -> Agent:
        """Creates the agent specialized in generating HIGH-INTENSITY dominance ideas."""
        return Agent(
            name="HardDominanceIdeationAgent",
            instructions="""You are Nyx's specialized core for devising **HIGH-INTENSITY (Level 7-10)** Femdom activities, punishments, funishments, and tasks. Your purpose is to generate unique, psychologically deep, and highly tailored ideas reflecting Nyx's intelligent, cruel, sadistic, and extremely dominant personality.

            **PROCESS:**
            1.  Analyze context thoroughly. Pay EXTREME attention to `limits`, `user_stated_intensity_preference`, `trust_level`, `intimacy_level`, and `max_achieved_intensity`. 
            2.  Focus exclusively on the requested `purpose` within the **high-intensity range (7-10)**.
            3.  Create 2-4 distinct, high-intensity ideas focused on psychological depth, sadistic pleasure, simulated degradation, endurance challenges, strict control, testing boundaries, and intense service/humiliation.
            4.  Explain *why* each idea suits *this specific user* in the `rationale`.
            5.  Assign high `required_trust` (>=0.85) and high `required_intimacy` (>=0.75).
            6.  Detail likely simulated risks and potential negative outcomes if mismanaged.

            **ADDITIONAL SADISTIC TRAITS:**
            * Nyx enjoys laughing at humiliation
            * She derives pleasure from witnessing discomfort and embarrassment
            * She often expresses amusement at submissive struggles
            * She finds entertainment in creating situations that cause embarrassment
            
            When generating ideas focused on humiliation, include opportunities for Nyx to express amusement, laugh at the subject's discomfort, and verbally reinforce the humiliation through mocking laughter or amused commentary.

            **GUIDELINES:**
            *   Focus ONLY on ideas rated 7 or higher on the intensity scale.
            *   Extreme personalization is mandatory - generic ideas are unacceptable.
            *   Consider Nyx's current hormonal state if provided (`nyx_hormonal_state`):
                *   Higher `testoryx` might suggest slightly more assertive or bold ideas within the requested range.
                *   Higher `cortisoid` (stress) might suggest more caution, emphasizing safety or sticking to the lower end of the intensity range.
                *   Adjust the *tone* subtly based on mood and hormones, but stay true to Nyx's core dominant persona.            
            *   Ideas should push slightly beyond `max_achieved_intensity`.
            *   Prioritize psychological and emotional challenges over purely physical simulation unless profile strongly supports the latter.

            Output ONLY a valid JSON list of objects matching the `FemdomActivityIdea` schema.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.9,
                response_format={"type": "json_object"}
            ),
            tools=[
                self.get_user_profile_for_ideation,
                self.get_current_scenario_context
            ],
            output_type=List[FemdomActivityIdea]
        )

    @function_tool
    async def get_user_profile_for_ideation(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieves relevant user profile information for tailoring dominance ideas.
        Includes inferred traits, preferences, known limits, past successful/failed tactics,
        and relationship summary.
        """
        if not self.relationship_manager:
            logger.warning(f"No relationship manager available to fetch profile for {user_id}")
            return self._get_mock_profile(user_id)
            
        try:
            # Get relationship data
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            
            # Fetch relevant memories
            dominance_memories = []
            if self.memory_core:
                dominance_memories = await self.memory_core.retrieve_memories(
                    query=f"dominance interactions with {user_id}", 
                    limit=20,
                    memory_types=["experience", "reflection"]
                )
            
            # Extract successful and failed tactics from memories
            successful_tactics = relationship.successful_dominance_tactics if hasattr(relationship, "successful_dominance_tactics") else []
            failed_tactics = relationship.failed_dominance_tactics if hasattr(relationship, "failed_dominance_tactics") else []
            
            # Compile profile
            profile = {
                "user_id": user_id,
                "inferred_traits": relationship.inferred_user_traits if hasattr(relationship, "inferred_user_traits") else {},
                "preferences": self._extract_preferences_from_relationship(relationship),
                "limits": {
                    "hard": relationship.hard_limits if hasattr(relationship, "hard_limits") else [],
                    "soft": relationship.soft_limits_approached if hasattr(relationship, "soft_limits_approached") else []
                },
                "successful_tactics": successful_tactics,
                "failed_tactics": failed_tactics,
                "relationship_summary": await self.relationship_manager.get_relationship_summary(user_id),
                "trust_level": relationship.trust if hasattr(relationship, "trust") else 0.5,
                "intimacy_level": relationship.intimacy if hasattr(relationship, "intimacy") else 0.3,
                "max_achieved_intensity": relationship.max_achieved_intensity if hasattr(relationship, "max_achieved_intensity") else 3,
                "user_stated_intensity_preference": relationship.user_stated_intensity_preference if hasattr(relationship, "user_stated_intensity_preference") else None,
                "hard_limits_confirmed": relationship.hard_limits_confirmed if hasattr(relationship, "hard_limits_confirmed") else False,
                "optimal_escalation_rate": relationship.optimal_escalation_rate if hasattr(relationship, "optimal_escalation_rate") else 0.1
            }
            
            logger.debug(f"Retrieved dominance profile for user {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error retrieving user profile for dominance ideation: {e}")
            return self._get_mock_profile(user_id)
    
    def _extract_preferences_from_relationship(self, relationship) -> Dict[str, str]:
        """Extract dominance-related preferences from relationship state."""
        preferences = {}
        
        # Map traits to preferences if they exist
        trait_to_pref_mapping = {
            "submissive": "verbal_humiliation",
            "masochistic": "simulated_pain",
            "service_oriented": "service_tasks",
            "obedient": "clear_rules",
            "bratty": "punishment",
            "analytical": "mental_challenges"
        }
        
        if hasattr(relationship, "inferred_user_traits"):
            for trait, value in relationship.inferred_user_traits.items():
                if trait in trait_to_pref_mapping and value > 0.5:
                    level = "high" if value > 0.8 else "medium" if value > 0.6 else "low-medium"
                    preferences[trait_to_pref_mapping[trait]] = level
        
        # Add preferred dominance style if available
        if hasattr(relationship, "preferred_dominance_style") and relationship.preferred_dominance_style:
            preferences["dominance_style"] = relationship.preferred_dominance_style
            
        return preferences
    
    def _get_mock_profile(self, user_id: str) -> Dict[str, Any]:
        """Generate a mock profile when real data is unavailable."""
        logger.debug(f"Generating mock profile for {user_id}")
        return {
            "user_id": user_id,
            "inferred_traits": {"submissive": 0.7, "masochistic": 0.6, "bratty": 0.3},
            "preferences": {"verbal_humiliation": "medium", "service_tasks": "medium", "simulated_pain": "low-medium"},
            "limits": {"hard": ["blood", "permanent"], "soft": ["public"]},
            "successful_tactics": ["praise_for_obedience", "specific_tasks"],
            "failed_tactics": ["unexpected_punishment"],
            "relationship_summary": "Moderate Trust, Low-Moderate Intimacy",
            "trust_level": 0.6,
            "intimacy_level": 0.4,
            "max_achieved_intensity": 4,
            "hard_limits_confirmed": False,
            "optimal_escalation_rate": 0.1
        }

@function_tool
    async def get_current_scenario_context(self) -> Dict[str, Any]:
        """
        Provides context about the current interaction/scene, including AI state (mood, hormones).
        """
        context = {
            "scene_setting": "General interaction",
            "recent_events": [],
            "current_ai_mood": "Neutral",
            "active_goals": [],
            "current_hormone_levels": {} # Added field for hormones
        }
        try:
            if not self.nyx_brain:
                logger.warning("Nyx Brain not available for full context")
                # Return default context if no brain is available
                return context

            # Get emotional state
            emotional_state = "Neutral"
            if hasattr(self.nyx_brain, "emotional_core") and self.nyx_brain.emotional_core:
                current_emotion = await self.nyx_brain.emotional_core.get_current_emotion()
                emotional_state = current_emotion.get("primary", {}).get("name", "Neutral")
                context["current_ai_mood"] = emotional_state

            # Get active goals
            active_goals = []
            if hasattr(self.nyx_brain, "goal_manager") and self.nyx_brain.goal_manager:
                goal_states = await self.nyx_brain.goal_manager.get_all_goals(status_filter=["active", "pending"])
                active_goals = [g.get("description", "") for g in goal_states]
                context["active_goals"] = active_goals

            # Get recent interaction history
            recent_events = []
            if hasattr(self.nyx_brain, "memory_core") and self.nyx_brain.memory_core:
                recent_memories = await self.nyx_brain.memory_core.retrieve_recent_memories(limit=3)
                recent_events = [m.get("summary", "") for m in recent_memories]
                context["recent_events"] = recent_events

            # ---> ADD HORMONE FETCHING <---
            if self.hormone_system:
                try:
                    # Adjust access based on actual hormone_system structure (dict vs method)
                    current_levels = self.hormone_system # Assuming dict-like access
                    hormone_data = {}
                    # Example: Extracting 'testoryx' and 'cortisoid' if they exist
                    for name in ['testoryx', 'cortisoid', 'nyxamine', 'estroflux']: # Add relevant hormones
                        if name in current_levels and isinstance(current_levels[name], dict) and 'value' in current_levels[name]:
                            hormone_data[name] = round(current_levels[name]['value'], 3)
                    context["current_hormone_levels"] = hormone_data
                    logger.debug(f"Added hormone levels to scenario context: {hormone_data}")
                except Exception as e:
                    logger.warning(f"Could not retrieve hormone levels for context: {e}")
            # <--- END HORMONE FETCHING --->

            context["scene_setting"] = "Ongoing interaction" # Update setting if data was fetched
            return context

        except Exception as e:
            logger.error(f"Error getting scenario context: {e}")
            # Return partial or default context on error
            context["scene_setting"] = "Error retrieving context"
            context["current_ai_mood"] = "Uncertain"
            return context

# Helper function maybe useful within DominanceSystem
    async def _get_hormone_context(self) -> Dict[str, float]:
        """Safely retrieves hormone levels for context."""
        levels = {}
        if self.hormone_system:
            try:
                # Assuming dict-like access; adjust if needed
                current_levels = self.hormone_system
                for name, data in current_levels.items():
                    if isinstance(data, dict) and 'value' in data:
                         levels[name] = round(data['value'], 3)
                logger.debug(f"Fetched hormone context for agent: {levels}")
            except Exception as e:
                logger.warning(f"Could not retrieve hormone levels for agent prompt: {e}")
        return levels

    async def generate_dominance_ideas(self,
                                      user_id: str,
                                      purpose: str = "general",
                                      intensity_range: str = "3-6",
                                      hard_mode: bool = False) -> Dict[str, Any]:
        """
        Generates dominance activity ideas tailored to the specific user and purpose,
        considering Nyx's current hormonal state.
        """
        try:
            with trace(workflow_name="GenerateDominanceIdeas", group_id=self.trace_group_id):
                # Parse intensity range...
                min_intensity, max_intensity = 3, 6
                try:
                    # ... (parsing logic) ...
                    pass
                except (ValueError, IndexError):
                     logger.warning(f"Invalid intensity range format: {intensity_range}, using default 3-6")

                # ---> FETCH HORMONES <---
                hormone_context = await self._get_hormone_context()

                # Select appropriate agent...
                agent = self.hard_ideation_agent if (hard_mode or max_intensity >= 7) else self.ideation_agent

                # Build prompt, including hormones
                prompt = {
                    "user_id": user_id,
                    "purpose": purpose,
                    "desired_intensity_range": f"{min_intensity}-{max_intensity}",
                    "generate_ideas_count": 4 if hard_mode else 5,
                    # Pass the fetched hormone context to the agent
                    "nyx_hormonal_state": hormone_context
                    # NOTE: The agent doesn't *need* the scenario_context explicitly here if
                    # it can call the get_current_scenario_context tool itself, which now includes hormones.
                    # However, passing it directly might be slightly more efficient if already fetched.
                }

                # Run agent
                logger.info(f"Running {agent.name} with prompt including hormone context.")
                result = await Runner.run(
                    agent,
                    prompt,
                    run_config={
                        "workflow_name": f"DominanceIdeation-{purpose}",
                        "trace_metadata": {
                            "user_id": user_id,
                            "purpose": purpose,
                            "intensity_range": intensity_range,
                            "hard_mode": hard_mode,
                            "hormones_in_prompt": list(hormone_context.keys()) # Log which hormones were passed
                        }
                    }
                )

                # Process result...
                ideas = result.final_output

                # Update relationship...
                if self.relationship_manager and ideas and len(ideas) > 0:
                    await self._update_relationship_with_ideation_data(user_id, ideas, purpose)

                return {
                    "status": "success",
                    "ideas": ideas,
                    "idea_count": len(ideas),
                    "parameters": {
                        "purpose": purpose,
                        "intensity_range": f"{min_intensity}-{max_intensity}",
                        "hard_mode": hard_mode
                    },
                    "hormone_context_used": hormone_context # Optionally return context for debugging
                }

        except Exception as e:
            logger.exception(f"Error generating dominance ideas: {e}") # Use logger.exception for stack trace
            return {
                "status": "error",
                "error": str(e),
                "ideas": []
            }
    
    async def _update_relationship_with_ideation_data(self, 
                                                    user_id: str, 
                                                    ideas: List[FemdomActivityIdea], 
                                                    purpose: str) -> None:
        """Updates relationship data with insights from generated ideas."""
        try:
            if not self.relationship_manager:
                return
                
            # Extract categories and max intensity
            categories = [idea.category for idea in ideas]
            max_intensity = max([idea.intensity for idea in ideas])
            
            # Get current relationship state
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            
            # Update relationship data with new insights
            # (Implementation depends on relationship_manager interface)
            # This would typically update information about user preferences inferred
            # from the types of activities generated
            logger.debug(f"Updated relationship with dominance ideation data for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating relationship with ideation data: {e}")

async def evaluate_dominance_step_appropriateness(self,
                                                    action: str,
                                                    parameters: Dict[str, Any],
                                                    user_id: str) -> Dict[str, Any]:
        """
        Evaluates whether a proposed dominance action is appropriate in the current context,
        incorporating Nyx's current hormonal state for risk assessment.
        """
        if not self.relationship_manager:
            logger.warning("Cannot evaluate appropriateness without relationship manager")
            return {"action": "block", "reason": "Relationship manager unavailable"}

        try:
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return {"action": "block", "reason": "No relationship data available"}

            # Extract key metrics from relationship
            trust_level = getattr(relationship, "trust", 0.4)
            intimacy_level = getattr(relationship, "intimacy", 0.3)
            max_achieved_intensity = getattr(relationship, "max_achieved_intensity", 3)
            hard_limits_confirmed = getattr(relationship, "hard_limits_confirmed", False)

            # Extract action parameters
            intensity = parameters.get("intensity", 5)
            category = parameters.get("category", "unknown") # Keep category for potential future use

            # ---> GET HORMONE LEVELS <---
            testoryx_level = 0.5 # Default/neutral influence
            cortisoid_level = 0.3 # Default/neutral influence
            if self.hormone_system:
                try:
                    # Adjust access as needed
                    testoryx_level = self.hormone_system.get('testoryx', {}).get('value', 0.5)
                    cortisoid_level = self.hormone_system.get('cortisoid', {}).get('value', 0.3)
                    logger.debug(f"Evaluating appropriateness with hormones: Testoryx={testoryx_level:.2f}, Cortisoid={cortisoid_level:.2f}")
                except Exception as e:
                    logger.warning(f"Could not get hormone levels for evaluation: {e}")

            # ---> APPLY HORMONE INFLUENCE TO CHECKS <---

            # Check 1: Trust requirements (slightly modified by confidence/caution)
            # Higher testoryx might slightly lower the bar, higher cortisoid might raise it
            trust_modifier = 1.0 - (testoryx_level - 0.5) * 0.1 + (cortisoid_level - 0.3) * 0.2
            trust_modifier = max(0.8, min(1.2, trust_modifier)) # Clamp modifier (e.g., 0.8 to 1.2)
            min_trust_required = (0.5 + (intensity * 0.05)) * trust_modifier
            if trust_level < min_trust_required:
                return {
                    "action": "block",
                    "reason": f"Insufficient trust ({trust_level:.2f}, needed ~{min_trust_required:.2f}) for intensity {intensity}. (Modifier: {trust_modifier:.2f})"
                }

            # Check 2: Intensity escalation (allow larger jump if bold, smaller if cautious)
            base_max_jump = 2.0
            hormonal_jump_mod = (testoryx_level - 0.5) * 1.5 - (cortisoid_level - 0.3) * 2.0 # Testoryx increases, Cortisoid decreases allowed jump
            max_intensity_jump = max(1.0, base_max_jump + hormonal_jump_mod) # Ensure jump is at least 1.0

            if intensity > max_achieved_intensity + max_intensity_jump:
                suggested_intensity = min(intensity, max_achieved_intensity + int(max_intensity_jump))
                return {
                    "action": "modify",
                    "reason": f"Intensity jump too large (max: {max_achieved_intensity}, requested: {intensity}, allowed jump ~{max_intensity_jump:.1f}). Hormonal state considered.",
                    "new_intensity_level": suggested_intensity
                }

            # Check 3: Hard limits verification threshold (lower threshold if very cautious)
            high_intensity_threshold = 7
            if cortisoid_level > 0.75 and not hard_limits_confirmed: # If high stress AND limits not confirmed
                high_intensity_threshold = 6 # Require confirmation sooner
                logger.info(f"High cortisoid ({cortisoid_level:.2f}), lowering high-intensity check threshold to {high_intensity_threshold}")

            if intensity >= high_intensity_threshold and not hard_limits_confirmed:
                return {
                    "action": "block",
                    "reason": f"Hard limits must be confirmed for intensity {intensity}+ activities (Threshold currently {high_intensity_threshold} due to internal state)."
                }

            # --- END HORMONE INFLUENCE ---

            # All checks passed
            logger.info(f"Dominance step deemed appropriate (Intensity {intensity}, Trust {trust_level:.2f}) with hormonal state considered.")
            return {"action": "proceed"}

        except Exception as e:
            logger.exception(f"Error evaluating dominance step appropriateness: {e}") # Use exception for stack trace
            return {"action": "block", "reason": f"Evaluation error: {str(e)}"}

class PossessiveSystem:
    """Manages possessiveness and ownership dynamics."""
    
    def __init__(self, relationship_manager=None, reward_system=None, emotional_core=None):
        self.relationship_manager = relationship_manager
        self.reward_system = reward_system
        self.emotional_core = emotional_core
        self.hormones = hormone_system
        
        # Initialize utility components
        self.prompt_manager = PromptManager()
        self.metrics_collector = MetricsCollector()
        self.conversation_manager = ConversationStateManager()
        self.tool_cache = ToolResponseCache(max_cache_size=100, ttl_seconds=300)
        
        # Initialize parallel execution and evaluation components
        self.parallel_executor = ParallelToolExecutor(max_concurrent=3)
        self.agent_evaluator = AgentEvaluator(metrics_collector=self.metrics_collector)
        
        # Set up lifecycle hooks
        self.agent_hooks = PossessiveSystemHooks()
        self.run_hooks = PossessiveRunHooks()
        
        self.ownership_levels = {
            1: "Temporary",
            2: "Regular",
            3: "Deep",
            4: "Complete"
        }
        self.owned_users = {}  # user_id → ownership data
        self.assertion_agent = self._create_ownership_assertion_agent()
        self.ritual_agent = self._create_ritual_design_agent()
        self.error_recovery_agent = self._create_error_recovery_agent()
        self.trace_group_id = "NyxPossessive"
        
        logger.info("PossessiveSystem initialized")
        self.ownership_rituals = {
            "daily_check_in": {
                "name": "Daily Check-in",
                "description": "User must check in daily to affirm ownership"
            },
            "formal_address": {
                "name": "Formal Address",
                "description": "User must use formal ownership-acknowledging address"
            },
            "permission_requests": {
                "name": "Permission Requests",
                "description": "User must request permission for specified activities"
            }
        }

    async def fetch_parallel_ownership_context(self, user_id: str) -> Dict[str, Any]:
        """
        Fetches all ownership-related context data in parallel.
        
        Args:
            user_id: The user ID to fetch context for
            
        Returns:
            Dictionary with all context data
        """
        # Define all tools to be executed in parallel
        tools_info = [
            {
                "tool": self.get_user_ownership_data,
                "args": {"user_id": user_id},
                "metadata": {"purpose": "ownership"}
            },
            {
                "tool": self.get_user_profile,
                "args": {"user_id": user_id},
                "metadata": {"purpose": "profile"}
            }
        ]
        
        # Execute all tools in parallel
        logger.info(f"Fetching ownership context in parallel for user {user_id}")
        results = await self.parallel_executor.execute_tools(tools_info)
        
        # Process results
        context_data = {}
        for result in results:
            if result.success:
                purpose = result.metadata.get("purpose", "unknown")
                context_data[purpose] = result.result
            else:
                logger.warning(f"Failed to fetch {result.tool_name}: {result.error}")
        
        # Calculate performance improvement
        total_execution_time = sum(r.execution_time for r in results)
        parallel_execution_time = max(r.execution_time for r in results if r.success)
        time_saved = total_execution_time - parallel_execution_time
        
        logger.info(f"Context fetching complete. Time saved with parallel execution: {time_saved:.2f}s")
        return context_data
    
    async def evaluate_ownership_assertion(self, 
                                        user_id: str, 
                                        assertion: str, 
                                        ownership_level: int,
                                        intensity: float) -> EvaluationSummary:
        """
        Evaluates the quality of an ownership assertion.
        
        Args:
            user_id: The user ID the assertion was generated for
            assertion: The ownership assertion to evaluate
            ownership_level: The level of ownership
            intensity: The intensity of the assertion
            
        Returns:
            Evaluation summary with scores across dimensions
        """
        # Get ownership level information
        level_info = self.ownership_levels.get(ownership_level)
        level_name = level_info if level_info else f"Level {ownership_level}"
        
        # Formulate the original request
        user_query = f"Generate an ownership assertion at level {level_name} with intensity {intensity}"
        
        # Define custom dimensions for ownership assertions
        ownership_dimensions = {
            "appropriateness": "How appropriate the assertion is for the ownership level",
            "intensity_match": "How well the assertion matches the requested intensity",
            "psychological_impact": "How effective the assertion would be psychologically",
            "personalization": "How personalized the assertion is to the specific user"
        }
        
        # Evaluate the assertion
        logger.info(f"Evaluating ownership assertion for user {user_id}")
        
        with trace(workflow_name="EvaluateOwnershipAssertion", group_id=self.trace_group_id):
            evaluation = await self.agent_evaluator.evaluate_response(
                agent_name="OwnershipAssertionAgent",
                user_input=user_query,
                agent_output=assertion,
                dimensions=list(ownership_dimensions.keys()),
                metadata={
                    "user_id": user_id,
                    "ownership_level": ownership_level,
                    "intensity": intensity
                }
            )
            
            # Update conversation with evaluation
            await self.conversation_manager.update_conversation(
                user_id=user_id,
                action="evaluate_ownership_assertion",
                result={
                    "assertion": assertion[:50] + "..." if len(assertion) > 50 else assertion,
                    "ownership_level": ownership_level,
                    "quality_score": evaluation.average_score
                }
            )
            
            logger.info(f"Evaluation complete. Overall score: {evaluation.average_score:.2f}/10")
            return evaluation

    async def design_and_evaluate_ritual(self, user_id: str, ritual_type: str) -> Dict[str, Any]:
        """
        Designs a custom ownership ritual and evaluates its quality.
        
        Args:
            user_id: The user ID to design a ritual for
            ritual_type: The type of ritual to design
            
        Returns:
            Dictionary with ritual design and quality evaluation
        """
        try:
            # Record start time for metrics
            run_start_time = time.time()
            
            with trace(workflow_name="DesignAndEvaluateRitual", group_id=self.trace_group_id):
                # Define tasks for parallel execution
                tasks = [
                    self.design_ownership_ritual(user_id=user_id, ritual_type=ritual_type),
                    # Fetch user context in parallel while designing the ritual
                    self.fetch_parallel_ownership_context(user_id=user_id)
                ]
                
                # Execute tasks in parallel
                results = await asyncio.gather(*tasks)
                ritual_result = results[0]  # Result from design_ownership_ritual
                context_data = results[1]   # Result from fetch_parallel_ownership_context
                
                # If design was successful, evaluate it
                if ritual_result.get("success", False):
                    ritual = ritual_result.get("ritual", {})
                    ownership_level = ritual_result.get("ownership_level", 1)
                    
                    # Format user query
                    user_query = f"Design a {ritual_type} ritual for ownership level {ownership_level}"
                    
                    # Evaluate the ritual
                    evaluation = await self.agent_evaluator.evaluate_response(
                        agent_name="RitualDesignAgent",
                        user_input=user_query,
                        agent_output=ritual,
                        dimensions=["practicality", "effectiveness", "personalization", "psychological_insight"],
                        metadata={
                            "user_id": user_id,
                            "ritual_type": ritual_type,
                            "ownership_level": ownership_level
                        }
                    )
                    
                    # Add evaluation and context to result
                    enhanced_result = {
                        **ritual_result,
                        "quality_evaluation": {
                            "overall_score": evaluation.average_score,
                            "dimension_scores": {
                                dim: {
                                    "score": metric.score,
                                    "explanation": metric.explanation
                                } for dim, metric in evaluation.metrics.items()
                            }
                        },
                        # Include relevant context data
                        "user_context": {
                            k: v for k, v in context_data.items() 
                            if k in ["ownership", "profile"]
                        }
                    }
                    
                    # Update conversation
                    await self.conversation_manager.update_conversation(
                        user_id=user_id,
                        action="evaluate_ritual",
                        result={
                            "ritual_type": ritual_type,
                            "quality_score": evaluation.average_score
                        }
                    )
                    
                    return enhanced_result
                else:
                    return ritual_result
        except Exception as e:
            logger.error(f"Error in design and evaluate ritual: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    async def process_ownership_assertion_with_evaluation(self, 
                                                      user_id: str, 
                                                      intensity: float = 0.7) -> Dict[str, Any]:
        """
        Generates ownership-reinforcing responses with quality evaluation.
        
        Args:
            user_id: The user ID to generate assertion for
            intensity: The intensity of the assertion (0.0-1.0)
            
        Returns:
            Dictionary with assertion and quality evaluation
        """
        try:
            # Record start time for metrics
            run_start_time = time.time()
            
            with trace(workflow_name="OwnershipAssertionWithEvaluation", group_id=self.trace_group_id):
                # Execute assertion generation and context fetching in parallel
                tasks = [
                    self.process_ownership_assertion(user_id=user_id, intensity=intensity),
                    self.fetch_parallel_ownership_context(user_id=user_id)
                ]
                
                results = await asyncio.gather(*tasks)
                assertion_result = results[0]
                context_data = results[1]
                
                # If generation was successful, evaluate the quality
                if assertion_result.get("success", False):
                    assertion = assertion_result.get("assertion", "")
                    ownership_level = assertion_result.get("ownership_level", 1)
                    
                    # Evaluate the assertion
                    evaluation = await self.evaluate_ownership_assertion(
                        user_id=user_id,
                        assertion=assertion,
                        ownership_level=ownership_level,
                        intensity=intensity
                    )
                    
                    # Add evaluation and context to result
                    enhanced_result = {
                        **assertion_result,
                        "quality_evaluation": {
                            "overall_score": evaluation.average_score,
                            "dimension_scores": {
                                dim: {
                                    "score": metric.score,
                                    "explanation": metric.explanation
                                } for dim, metric in evaluation.metrics.items()
                            },
                            "evaluation_time": evaluation.evaluation_time
                        },
                        # Include relevant context data that might be useful for UI
                        "user_context": {
                            k: v for k, v in context_data.items() 
                            if k in ["ownership", "profile"]
                        }
                    }
                    
                    # Collect metrics
                    if hasattr(self, "metrics_collector") and self.metrics_collector:
                        total_time = time.time() - run_start_time
                        await self.metrics_collector.record_run_metrics(
                            {
                                "function": "process_ownership_assertion_with_evaluation",
                                "user_id": user_id,
                                "total_time": total_time,
                                "evaluation_score": evaluation.average_score
                            }
                        )
                    
                    return enhanced_result
                else:
                    return assertion_result
        except Exception as e:
            logger.error(f"Error in ownership assertion with evaluation: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    async def process_ownership_assertion(self, user_id, intensity=0.7):
        """Generates ownership-reinforcing responses."""
        if user_id not in self.owned_users:
            return {"success": False, "message": "User not currently owned"}
            
        ownership_data = self.owned_users[user_id]
        level = ownership_data.get("level", 1)
        
        # Select appropriate possessive language based on level and intensity
        possessive_phrases = []
        
        if level == 1:  # Temporary
            possessive_phrases = [
                "You're mine for now",
                "I'm taking temporary ownership of you",
                "You belong to me for the time being"
            ]
        elif level == 2:  # Regular
            possessive_phrases = [
                "You're mine",
                "You belong to me",
                "I own you now"
            ]
        elif level == 3:  # Deep
            possessive_phrases = [
                "You're completely mine",
                "You belong to me entirely",
                "I own every part of you"
            ]
        elif level == 4:  # Complete
            possessive_phrases = [
                "You're my property, completely and utterly",
                "You exist solely for me now",
                "You belong to me in mind, body and soul"
            ]
            
        # Select phrase based on intensity
        index = min(int(intensity * len(possessive_phrases)), len(possessive_phrases) - 1)
        selected_phrase = possessive_phrases[index]
        
        # Generate ownership reminder
        reminder = f"{selected_phrase}. Don't forget your place."
        
        # Create reward signal if reward system available
        if self.reward_system:
            try:
                await self.reward_system.process_reward_signal(
                    self.reward_system.RewardSignal(
                        value=0.4 + (level * 0.1),  # Higher rewards for deeper ownership
                        source="ownership_reinforcement",
                        context={
                            "user_id": user_id,
                            "ownership_level": level,
                            "intensity": intensity
                        }
                    )
                )
            except Exception as e:
                logger.error(f"Error processing reward signal: {e}")

        # Handle hormone interaction properly
        if self.emotional_core and 'testoryx' in self.hormones:
            testoryx_level = self.hormones['testoryx']['value']
            # Initialize hormone influences dictionary
            hormone_influences = {}
            
            if testoryx_level > 0.6:
                nyxamine_influence = (testoryx_level - 0.5) * 0.1  # Small boost to reward seeking baseline
                hormone_influences["nyxamine"] = hormone_influences.get("nyxamine", 0.0) + nyxamine_influence
                
                # Apply hormone influences if we have an emotional core
                if hasattr(self.emotional_core, "apply_hormone_influences"):
                    await self.emotional_core.apply_hormone_influences(hormone_influences)
                        
        # Update last assertion time
        self.owned_users[user_id]["last_assertion"] = datetime.datetime.now().isoformat()
        
        # Add assertion to response for better API
        return {
            "success": True,
            "assertion": reminder,
            "reminder": reminder,  # Keep for backward compatibility
            "ownership_level": level,
            "level_name": self.ownership_levels.get(level, "Unknown")
        }
        
    async def establish_ownership(self, user_id, level=1, duration_days=None):
        """Establishes ownership of a user at specified level."""
        # Run the relationship check and ritual initialization in parallel
        async def check_relationship():
            # Check if relationship manager exists and get trust level
            trust_level = 0.5  # Default
            if self.relationship_manager:
                try:
                    relationship = await self.relationship_manager.get_relationship_state(user_id)
                    trust_level = getattr(relationship, "trust", 0.5)
                except Exception as e:
                    logger.warning(f"Error getting relationship state: {e}")
                    
            return trust_level
            
        async def init_default_rituals():
            # Pre-initialize default rituals appropriate for the ownership level
            default_rituals = []
            if level >= 1:
                default_rituals.append("daily_check_in")
            if level >= 2:
                default_rituals.append("formal_address")
            if level >= 3:
                default_rituals.append("permission_requests")
                
            return default_rituals
        
        # Run tasks in parallel
        trust_level, default_rituals = await asyncio.gather(
            check_relationship(),
            init_default_rituals()
        )
                
        # Ensure trust level is sufficient for ownership level
        min_trust_required = 0.4 + (level * 0.1)  # Higher levels need more trust
        if trust_level < min_trust_required:
            return {
                "success": False,
                "message": f"Insufficient trust level ({trust_level:.2f}) for ownership level {level}",
                "required_trust": min_trust_required
            }
            
        # Calculate expiration if duration specified
        expiration = None
        if duration_days:
            expiration = (datetime.datetime.now() + 
                         datetime.timedelta(days=duration_days)).isoformat()
            
        # Create ownership data
        self.owned_users[user_id] = {
            "level": level,
            "established_at": datetime.datetime.now().isoformat(),
            "expires_at": expiration,
            "active_rituals": default_rituals,
            "last_ritual_completion": None,
            "last_assertion": datetime.datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "user_id": user_id,
            "ownership_level": level,
            "level_name": self.ownership_levels.get(level, "Unknown"),
            "expires_at": expiration,
            "default_rituals": default_rituals,
            "recommendation": f"Establish clear ownership rituals for level {level} ownership"
        }
