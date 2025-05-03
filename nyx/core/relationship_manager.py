# nyx/core/relationship_manager.py

import logging
import datetime
import math
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Set
from pydantic import BaseModel, Field

from openai import AsyncOpenAI
from agents import Agent, Runner, ModelSettings, trace, function_tool, GuardrailFunctionOutput, InputGuardrail

logger = logging.getLogger(__name__)

class RelationshipState(BaseModel):
    """Represents the state of Nyx's relationship with a specific user."""
    user_id: str
    trust: float = Field(0.5, ge=0.0, le=1.0, description="Level of trust (0=None, 1=Complete)")
    familiarity: float = Field(0.1, ge=0.0, le=1.0, description="How well Nyx 'knows' the user (0=Stranger, 1=Intimate)")
    intimacy: float = Field(0.1, ge=0.0, le=1.0, description="Level of emotional closeness/vulnerability shared")
    conflict: float = Field(0.1, ge=0.0, le=1.0, description="Level of unresolved conflict/tension")
    dominance_balance: float = Field(0.0, ge=-1.0, le=1.0, description="Perceived dominance (-1=User Dominant, 0=Balanced, 1=Nyx Dominant)")
    positive_interaction_score: float = Field(0.0, description="Cumulative positive interaction value")
    negative_interaction_score: float = Field(0.0, description="Cumulative negative interaction value")
    interaction_count: int = 0
    last_interaction_time: Optional[datetime.datetime] = None
    key_memories: List[str] = Field(default_factory=list, description="IDs of key memories shared/related to this user")
    inferred_user_traits: Dict[str, float] = Field(default_factory=dict, description="Nyx's inferred traits about the user")
    shared_secrets_level: float = Field(0.0, ge=0.0, le=1.0, description="How much sensitive info shared")
    
    # Dominance-related fields
    current_dominance_intensity: float = Field(0.0, ge=0.0, le=1.0, description="Current level of expressed dominance")
    max_achieved_intensity: float = Field(0.0, ge=0.0, le=1.0, description="Highest intensity level successfully reached")
    failed_escalation_attempts: int = 0
    successful_dominance_tactics: List[str] = Field(default_factory=list)
    failed_dominance_tactics: List[str] = Field(default_factory=list)
    preferred_dominance_style: Optional[str] = None
    optimal_escalation_rate: float = Field(0.05, description="Learned optimal step size for intensity increase")
    user_stated_intensity_preference: Optional[Union[int, str]] = None
    hard_limits_confirmed: bool = Field(False)
    hard_limits: List[str] = Field(default_factory=list)
    soft_limits_approached: List[str] = Field(default_factory=list)
    soft_limits_crossed_successfully: List[str] = Field(default_factory=list)

class InteractionRecord(BaseModel):
    """A record of a significant interaction with a user."""
    user_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    interaction_type: str  # e.g., "conversation", "dominance_session", "emotional_exchange"
    valence: float = Field(0.0, ge=-1.0, le=1.0)  # Overall positivity/negativity
    trust_impact: float = 0.0  # How this impacted trust
    intimacy_impact: float = 0.0  # How this impacted intimacy
    dominance_change: float = 0.0  # How this affected dominance balance
    summary: str  # Brief summary of the interaction
    memory_ids: List[str] = Field(default_factory=list)  # Related memory IDs
    emotion_tags: List[str] = Field(default_factory=list)  # Key emotions observed
    learned_preferences: Dict[str, Any] = Field(default_factory=dict)  # Any preferences learned

class RelationshipManager:
    """Manages Nyx's relationship states with different users."""

    def __init__(self, memory_orchestrator=None, emotional_core=None):
        self.memory_orchestrator = memory_orchestrator  # For storing/retrieving relationship-linked memories
        self.relationships: Dict[str, RelationshipState] = {}  # user_id -> RelationshipState
        self.interaction_history: Dict[str, List[InteractionRecord]] = {}  # user_id -> list of interactions
        self.max_interactions_per_user = 50  # Maximum interactions to keep per user
        self._lock = asyncio.Lock()  # Lock for thread-safe operations
        
        # Create trait/preference inference agent
        self.trait_agent = self._create_trait_inference_agent()
        
        # Weights for updating relationship metrics
        self.update_weights = {
            "positive_valence": 0.05,  # How much positive emotion increases trust/intimacy
            "negative_valence": 0.08,  # How much negative emotion increases conflict/decreases trust
            "shared_experience": 0.03,  # Impact of sharing an experience
            "shared_reflection": 0.05,  # Impact of sharing a reflection
            "goal_success_user": 0.04,  # Impact of success on user-related goal
            "goal_failure_user": 0.06,  # Impact of failure on user-related goal
            "user_feedback_positive": 0.1,
            "user_feedback_negative": 0.15,
            "time_decay_factor": 0.998,  # Slow decay per day for trust/intimacy if no interaction
        }
        
        logger.info("RelationshipManager initialized")

    def _create_trait_inference_agent(self) -> Optional[Agent]:
        """Creates an agent for inferring user traits from interactions."""
        try:
            return Agent(
                name="User Trait Inference Agent",
                instructions="""You analyze user interactions to infer traits, preferences, and relationship characteristics.
                Based on interaction history, emotional responses, and explicit/implicit signals, update Nyx's understanding of a user.
                
                When analyzing interactions, pay attention to:
                1. Linguistic patterns suggesting personality traits
                2. Emotional responses to different stimuli
                3. Explicit preferences or boundaries stated
                4. Implicit preferences revealed through engagement patterns
                5. Reactions to different interaction styles (directive vs. supportive, etc.)
                
                Your output should be a JSON object containing:
                - Updated trait scores (0.0-1.0) for relevant traits
                - New or modified preferences with confidence levels
                - Identified boundaries or limits
                - Recommended relationship development approaches
                
                Focus on evidence-based inference rather than stereotyping.
                """,
                model="gpt-4o",
                model_settings=ModelSettings(
                    temperature=0.3,
                ),
                output_type=Dict[str, Any]
            )
        except Exception as e:
            logger.error(f"Error creating trait inference agent: {e}")
            return None

    async def get_or_create_relationship_internal(self, user_id: str) -> Dict[str, Any]:
        """Gets the relationship state for a user, creating it if it doesn't exist."""
        if user_id not in self.relationships:
            logger.info(f"Creating new relationship profile for user '{user_id}'")
            self.relationships[user_id] = RelationshipState(user_id=user_id)
            self.interaction_history[user_id] = []
            
        state = self.relationships[user_id]
        return state.model_dump()

    async def get_all_relationship_ids_internal(self) -> List[str]:
        """Returns a list of all user IDs with relationship states."""
        async with self._lock:
            return list(self.relationships.keys())

    async def get_relationship_state_internal(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Gets the current relationship state for a user."""
        # Apply time decay before returning if significant time has passed
        async with self._lock:
            if user_id not in self.relationships:
                logger.info(f"Creating new relationship for user {user_id} on first query")
                await self.get_or_create_relationship_internal(user_id)
                
            state = self.relationships[user_id]
            now = datetime.datetime.now()
            
            if state.last_interaction_time:
                days_since = (now - state.last_interaction_time).total_seconds() / (3600 * 24)
                if days_since > 0.5:  # Only apply decay if significant time passed
                    time_decay = math.pow(self.update_weights["time_decay_factor"], days_since)
                    state.trust *= time_decay
                    state.intimacy *= time_decay
                    state.familiarity *= time_decay
                    
                    # Clamp again after decay
                    state.trust = max(0.0, min(1.0, state.trust))
                    state.familiarity = max(0.0, min(1.0, state.familiarity))
                    state.intimacy = max(0.0, min(1.0, state.intimacy))
                    
                    # Don't decay conflict or dominance balance this way
                    state.last_interaction_time = now  # Update time to prevent repeated decay on consecutive gets
            
            return state.model_dump()

    @function_tool
    async def get_relationship_summary(self, user_id: str) -> str:
        """Provides a brief textual summary of the relationship."""
        state_dict = await self.get_relationship_state(user_id)
        if not state_dict: 
            return "No relationship data available."
        
        state = RelationshipState(**state_dict)
        
        summary = f"Relationship with {user_id}: "
        
        # Trust description
        if state.trust > 0.8: 
            summary += "Very High Trust. "
        elif state.trust > 0.6: 
            summary += "High Trust. "
        elif state.trust > 0.4: 
            summary += "Moderate Trust. "
        else: 
            summary += "Low Trust. "

        # Familiarity description  
        if state.familiarity > 0.7: 
            summary += "Very Familiar. "
        elif state.familiarity > 0.4: 
            summary += "Familiar. "
        else: 
            summary += "Getting Acquainted. "

        # Intimacy description
        if state.intimacy > 0.6: 
            summary += "High Intimacy. "
        elif state.intimacy > 0.3: 
            summary += "Moderate Intimacy. "
        else: 
            summary += "Low Intimacy. "

        # Conflict note if significant
        if state.conflict > 0.5: 
            summary += f"Notable Conflict (Level: {state.conflict:.2f}). "

        # Dominance balance
        dom = state.dominance_balance
        if dom > 0.4: 
            summary += "Nyx is dominant. "
        elif dom < -0.4: 
            summary += "User is dominant. "
        else: 
            summary += "Balanced dynamic. "
            
        # Include interaction count
        summary += f"({state.interaction_count} interactions)"

        return summary.strip()

    async def get_interaction_history_internal(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interaction records for a user."""
        async with self._lock:
            if user_id not in self.interaction_history:
                return []
                
            interactions = self.interaction_history[user_id][-limit:]
            
            # Convert to dict for return
            return [
                {
                    "timestamp": interaction.timestamp.isoformat(),
                    "type": interaction.interaction_type,
                    "valence": interaction.valence,
                    "trust_impact": interaction.trust_impact,
                    "intimacy_impact": interaction.intimacy_impact,
                    "dominance_change": interaction.dominance_change,
                    "summary": interaction.summary,
                    "emotions": interaction.emotion_tags
                }
                for interaction in interactions
            ]
    
    
    @function_tool
    async def get_or_create_relationship(self, user_id: str) -> Dict[str, Any]:
        """Gets the relationship state for a user, creating it if it doesn't exist."""
        return await self.get_or_create_relationship_internal(user_id)

    async def update_relationship_on_interaction(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Updates the relationship state based on a completed interaction."""
        if not user_id:
            return {"status": "error", "message": "Missing user ID"}

        async with self._lock:
            state_dict = await self.get_or_create_relationship(user_id)
            state = RelationshipState(**state_dict)
            now = datetime.datetime.now()

            # Apply time decay since last interaction
            if state.last_interaction_time:
                days_since = (now - state.last_interaction_time).total_seconds() / (3600 * 24)
                time_decay = math.pow(self.update_weights["time_decay_factor"], days_since)
                state.trust *= time_decay
                state.intimacy *= time_decay
                state.familiarity *= time_decay

            # Update interaction count and timestamp
            state.interaction_count += 1
            state.last_interaction_time = now

            # Extract emotional context
            emo_context = interaction_data.get("emotional_context", {})
            valence = emo_context.get("valence", 0.0)
            arousal = emo_context.get("arousal", 0.5)
            primary_emotion = emo_context.get("primary_emotion", {}).get("name", "neutral")
            
            # Track impacts for interaction record
            trust_impact = 0.0
            intimacy_impact = 0.0
            dominance_impact = 0.0

            # Update based on emotional valence
            if valence > 0.3:  # Positive interaction
                trust_impact = self.update_weights["positive_valence"] * valence
                state.positive_interaction_score += valence
                state.trust = min(1.0, state.trust + trust_impact)
                state.familiarity = min(1.0, state.familiarity + 0.01)  # Familiarity increases slowly
                
                # Different emotions affect intimacy differently
                if primary_emotion in ["Joy", "Trust", "Love"]:
                    intimacy_impact = 0.02 * valence
                    state.intimacy = min(1.0, state.intimacy + intimacy_impact)
                    
            elif valence < -0.3:  # Negative interaction
                trust_impact = -self.update_weights["negative_valence"] * abs(valence)
                state.negative_interaction_score += abs(valence)
                state.trust = max(0.0, state.trust + trust_impact)  # Note: trust_impact is negative here
                state.conflict = min(1.0, state.conflict + 0.05 * abs(valence))
                
                if primary_emotion in ["Anger", "Disgust", "Fear"]:
                    intimacy_impact = -0.03 * abs(valence)
                    state.intimacy = max(0.0, state.intimacy + intimacy_impact)  # Note: intimacy_impact is negative

            # Update based on experience/reflection sharing
            if interaction_data.get("shared_experience"):
                state.familiarity = min(1.0, state.familiarity + 0.02)
                intimacy_impact += self.update_weights["shared_experience"]
                state.intimacy = min(1.0, state.intimacy + self.update_weights["shared_experience"])
                
            if interaction_data.get("shared_reflection"):
                state.familiarity = min(1.0, state.familiarity + 0.03)
                intimacy_impact += self.update_weights["shared_reflection"]
                state.intimacy = min(1.0, state.intimacy + self.update_weights["shared_reflection"])

            # Update based on explicit user feedback
            feedback = interaction_data.get("user_feedback")
            if feedback:
                rating = feedback.get("rating", 0.5)
                if feedback.get("type") == "positive":
                    feedback_impact = self.update_weights["user_feedback_positive"] * rating
                    trust_impact += feedback_impact
                    state.trust = min(1.0, state.trust + feedback_impact)
                    state.positive_interaction_score += rating
                elif feedback.get("type") == "negative":
                    feedback_impact = -self.update_weights["user_feedback_negative"] * rating
                    trust_impact += feedback_impact
                    state.trust = max(0.0, state.trust + feedback_impact)  # Note: feedback_impact is negative
                    state.negative_interaction_score += rating
                    state.conflict = min(1.0, state.conflict + 0.08 * rating)

            # Update based on goal outcomes
            goal_outcome = interaction_data.get("goal_outcome")
            if goal_outcome:
                status = goal_outcome.get("status")
                priority = goal_outcome.get("priority", 0.5)
                if status == "completed":
                    goal_impact = self.update_weights["goal_success_user"] * priority
                    trust_impact += goal_impact
                    state.trust = min(1.0, state.trust + goal_impact)
                    state.familiarity = min(1.0, state.familiarity + 0.01 * priority)
                elif status == "failed":
                    goal_impact = -self.update_weights["goal_failure_user"] * priority
                    trust_impact += goal_impact
                    state.trust = max(0.0, state.trust + goal_impact)  # Note: goal_impact is negative
                    state.conflict = min(1.0, state.conflict + 0.04 * priority)

            # Update dominance balance based on interaction style
            nyx_style = interaction_data.get("nyx_response_style", "")
            user_style = interaction_data.get("user_input_style", "")
            if "dominant" in nyx_style and "submissive" in user_style:
                dominance_impact = 0.05
                state.dominance_balance = min(1.0, state.dominance_balance + dominance_impact)
            elif "submissive" in nyx_style and "dominant" in user_style:
                dominance_impact = -0.05
                state.dominance_balance = max(-1.0, state.dominance_balance + dominance_impact)
            elif "dominant" in nyx_style and "dominant" in user_style:
                state.conflict = min(1.0, state.conflict + 0.02)  # Dominance clash -> conflict
            else:
                # Gradual drift toward neutral
                state.dominance_balance *= 0.98

            # Handle dominance-specific updates
            if "dominance" in interaction_data:
                dom_data = interaction_data["dominance"]
                # Update current intensity level
                if "intensity_level" in dom_data:
                    state.current_dominance_intensity = dom_data["intensity_level"]
                    
                # Record successful tactics
                if "successful_tactic" in dom_data:
                    tactic = dom_data["successful_tactic"]
                    if tactic and tactic not in state.successful_dominance_tactics:
                        state.successful_dominance_tactics.append(tactic)
                        
                # Record failed tactics
                if "failed_tactic" in dom_data:
                    tactic = dom_data["failed_tactic"]
                    if tactic and tactic not in state.failed_dominance_tactics:
                        state.failed_dominance_tactics.append(tactic)
                        
                # Update max achieved intensity if current exceeds previous max
                if state.current_dominance_intensity > state.max_achieved_intensity:
                    state.max_achieved_intensity = state.current_dominance_intensity
                    
                # Track escalation attempts
                if dom_data.get("escalation_attempt", False) and dom_data.get("escalation_success", True) == False:
                    state.failed_escalation_attempts += 1
                    
                # Update limits if present
                if "hard_limits" in dom_data:
                    for limit in dom_data["hard_limits"]:
                        if limit and limit not in state.hard_limits:
                            state.hard_limits.append(limit)
                            
                if "soft_limits_approached" in dom_data:
                    for limit in dom_data["soft_limits_approached"]:
                        if limit and limit not in state.soft_limits_approached:
                            state.soft_limits_approached.append(limit)
                            
                if "soft_limits_crossed" in dom_data:
                    for limit in dom_data["soft_limits_crossed"]:
                        if limit and limit not in state.soft_limits_crossed_successfully:
                            state.soft_limits_crossed_successfully.append(limit)
                            
                # Update user preference if explicitly stated
                if "user_stated_intensity_preference" in dom_data:
                    state.user_stated_intensity_preference = dom_data["user_stated_intensity_preference"]
                    
                # Update hard limits confirmed flag
                if dom_data.get("hard_limits_confirmed", False):
                    state.hard_limits_confirmed = True

            # Link key memories
            memory_id = interaction_data.get("memory_id")
            significance = interaction_data.get("significance", 0)
            if memory_id and significance > 7:
                if memory_id not in state.key_memories:
                    state.key_memories.append(memory_id)
                    # Keep only last N key memories
                    state.key_memories = state.key_memories[-20:]

            # Clamp all values
            state.trust = max(0.0, min(1.0, state.trust))
            state.familiarity = max(0.0, min(1.0, state.familiarity))
            state.intimacy = max(0.0, min(1.0, state.intimacy))
            state.conflict = max(0.0, min(1.0, state.conflict))
            state.dominance_balance = max(-1.0, min(1.0, state.dominance_balance))
            
            # Create interaction record
            interaction_record = InteractionRecord(
                user_id=user_id,
                interaction_type=interaction_data.get("interaction_type", "conversation"),
                valence=valence,
                trust_impact=trust_impact,
                intimacy_impact=intimacy_impact,
                dominance_change=dominance_impact,
                summary=interaction_data.get("summary", "Interaction occurred"),
                memory_ids=[memory_id] if memory_id else [],
                emotion_tags=[primary_emotion] if primary_emotion != "neutral" else []
            )
            
            # Add to interaction history
            self.interaction_history[user_id].append(interaction_record)
            if len(self.interaction_history[user_id]) > self.max_interactions_per_user:
                self.interaction_history[user_id] = self.interaction_history[user_id][-self.max_interactions_per_user:]
            
            # Update the relationship in the dictionary
            self.relationships[user_id] = state

            # Update inferred traits if enough interactions have occurred
            if state.interaction_count % 5 == 0 or "dominance" in interaction_data:
                await self._update_inferred_traits(user_id)

            logger.debug(f"Updated relationship for user '{user_id}': Trust={state.trust:.2f}, Familiarity={state.familiarity:.2f}, Intimacy={state.intimacy:.2f}, Conflict={state.conflict:.2f}")
            
            return {
                "status": "success",
                "user_id": user_id,
                "trust": state.trust,
                "intimacy": state.intimacy,
                "familiarity": state.familiarity,
                "conflict": state.conflict,
                "dominance_balance": state.dominance_balance,
                "trust_impact": trust_impact,
                "intimacy_impact": intimacy_impact,
                "dominance_impact": dominance_impact
            }

    async def _update_inferred_traits(self, user_id: str) -> Dict[str, Any]:
        """Uses an agent to update inferred traits from interaction history."""
        if not self.trait_agent:
            logger.warning("Trait inference agent not available")
            return {"status": "error", "reason": "trait_agent_unavailable"}
            
        async with self._lock:
            state_dict = await self.get_or_create_relationship(user_id)
            state = RelationshipState(**state_dict)
            interactions = self.interaction_history.get(user_id, [])
            
            if not interactions:
                return {"status": "no_interactions"}
                
        try:
            # Prepare the data for inference
            recent_interactions = interactions[-10:]  # Use the last 10 interactions at most
            
            context = {
                "user_id": user_id,
                "relationship_state": {
                    "trust": state.trust,
                    "intimacy": state.intimacy,
                    "familiarity": state.familiarity,
                    "dominance_balance": state.dominance_balance,
                    "interaction_count": state.interaction_count
                },
                "current_traits": state.inferred_user_traits,
                "recent_interactions": [
                    {
                        "timestamp": interaction.timestamp.isoformat(),
                        "type": interaction.interaction_type,
                        "valence": interaction.valence,
                        "summary": interaction.summary,
                        "emotions": interaction.emotion_tags
                    }
                    for interaction in recent_interactions
                ],
                "dominance_data": {
                    "current_intensity": state.current_dominance_intensity,
                    "max_achieved_intensity": state.max_achieved_intensity,
                    "successful_tactics": state.successful_dominance_tactics,
                    "failed_tactics": state.failed_dominance_tactics,
                    "hard_limits": state.hard_limits,
                    "soft_limits_approached": state.soft_limits_approached,
                    "soft_limits_crossed": state.soft_limits_crossed_successfully
                }
            }
            
            # Run the trait inference agent
            with trace(workflow_name="InferUserTraits", group_id="RelationshipManager"):
                result = await Runner.run(
                    self.trait_agent,
                    json.dumps(context),
                    run_config={
                        "workflow_name": "TraitInference",
                        "trace_metadata": {"user_id": user_id}
                    }
                )
                
                # Process the result
                inference = result.final_output
                
                if not isinstance(inference, dict):
                    raise ValueError(f"Expected dict from trait agent, got {type(inference)}")
                
                async with self._lock:
                    # Get current state again in case it was updated
                    state_dict = await self.get_or_create_relationship(user_id)
                    state = RelationshipState(**state_dict)
                    
                    # Update the traits
                    if "updated_traits" in inference:
                        updated_traits = inference["updated_traits"]
                        # Merge with existing traits
                        for trait, value in updated_traits.items():
                            # Smooth the update - don't change too drastically
                            current = state.inferred_user_traits.get(trait, 0.5)
                            # 70% weight to new value, 30% to existing
                            state.inferred_user_traits[trait] = current * 0.3 + value * 0.7
                            
                    # Update dominance preferences if present
                    if "dominance_preferences" in inference:
                        dom_prefs = inference["dominance_preferences"]
                        
                        if "preferred_style" in dom_prefs and dom_prefs["preferred_style"]:
                            state.preferred_dominance_style = dom_prefs["preferred_style"]
                            
                        if "optimal_escalation_rate" in dom_prefs:
                            rate = dom_prefs["optimal_escalation_rate"]
                            if isinstance(rate, (int, float)) and 0.01 <= rate <= 0.5:
                                state.optimal_escalation_rate = rate
                    
                    # Update the relationship state in the dictionary
                    self.relationships[user_id] = state
                
                logger.info(f"Updated trait inference for user {user_id}")
                return {
                    "status": "success",
                    "updated_traits_count": len(inference.get("updated_traits", {})),
                    "traits": state.inferred_user_traits
                }
                
        except Exception as e:
            logger.error(f"Error updating inferred traits for user {user_id}: {e}")
            return {"status": "error", "reason": str(e)}

    @function_tool
    async def get_all_relationship_ids(self) -> List[str]:
        """Returns a list of all user IDs with relationship states."""
        return await self.get_all_relationship_ids_internal()

    @function_tool
    async def get_relationship_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Gets the current relationship state for a user."""
        return await self.get_relationship_state_internal(user_id)

    @function_tool
    async def get_relationship_summary(self, user_id: str) -> str:
        """Provides a brief textual summary of the relationship."""
        return await self.get_relationship_summary_internal(user_id)

    
    @function_tool
    async def get_interaction_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interaction records for a user."""
        return await self.get_interaction_history_internal(user_id, limit)
    
    async def merge_relationship_data(self, source_user_id: str, target_user_id: str) -> Dict[str, Any]:
        """
        Merge relationship data from source user into target user.
        Useful when discovering two user_ids belong to the same person.
        """
        async with self._lock:
            if source_user_id not in self.relationships or target_user_id not in self.relationships:
                return {"status": "error", "message": "Both source and target users must exist"}
            
            source = self.relationships[source_user_id]
            target = self.relationships[target_user_id]
            
            # Merge basic metrics with weighting by interaction count
            source_weight = source.interaction_count / (source.interaction_count + target.interaction_count)
            target_weight = 1.0 - source_weight
            
            # Update metrics
            target.trust = target.trust * target_weight + source.trust * source_weight
            target.familiarity = target.familiarity * target_weight + source.familiarity * source_weight
            target.intimacy = target.intimacy * target_weight + source.intimacy * source_weight
            target.conflict = max(target.conflict, source.conflict)  # Take the higher conflict level
            
            # Dominance is trickier - use the more established one if significantly different
            if abs(target.dominance_balance - source.dominance_balance) > 0.3:
                if source.interaction_count > target.interaction_count:
                    target.dominance_balance = source.dominance_balance
            else:
                # Otherwise blend them
                target.dominance_balance = (target.dominance_balance * target_weight + 
                                          source.dominance_balance * source_weight)
            
            # Merge interaction scores
            target.positive_interaction_score += source.positive_interaction_score
            target.negative_interaction_score += source.negative_interaction_score
            target.interaction_count += source.interaction_count
            
            # Take the most recent interaction time
            if source.last_interaction_time and (not target.last_interaction_time or 
                                              source.last_interaction_time > target.last_interaction_time):
                target.last_interaction_time = source.last_interaction_time
            
            # Merge key memories
            for memory_id in source.key_memories:
                if memory_id not in target.key_memories:
                    target.key_memories.append(memory_id)
            
            # Merge user traits - take the higher confidence ones
            for trait, value in source.inferred_user_traits.items():
                if trait not in target.inferred_user_traits or abs(target.inferred_user_traits[trait] - 0.5) < abs(value - 0.5):
                    target.inferred_user_traits[trait] = value
            
            # Merge dominance-related fields
            target.max_achieved_intensity = max(target.max_achieved_intensity, source.max_achieved_intensity)
            target.failed_escalation_attempts += source.failed_escalation_attempts
            
            # Merge tactics lists
            for tactic in source.successful_dominance_tactics:
                if tactic not in target.successful_dominance_tactics:
                    target.successful_dominance_tactics.append(tactic)
                    
            for tactic in source.failed_dominance_tactics:
                if tactic not in target.failed_dominance_tactics:
                    target.failed_dominance_tactics.append(tactic)
            
            # Merge limits
            for limit in source.hard_limits:
                if limit not in target.hard_limits:
                    target.hard_limits.append(limit)
                    
            for limit in source.soft_limits_approached:
                if limit not in target.soft_limits_approached:
                    target.soft_limits_approached.append(limit)
                    
            for limit in source.soft_limits_crossed_successfully:
                if limit not in target.soft_limits_crossed_successfully:
                    target.soft_limits_crossed_successfully.append(limit)
            
            # If either had hard limits confirmed, consider the merged one confirmed
            target.hard_limits_confirmed = target.hard_limits_confirmed or source.hard_limits_confirmed
            
            # Prefer explicit user preference if available
            if source.user_stated_intensity_preference and not target.user_stated_intensity_preference:
                target.user_stated_intensity_preference = source.user_stated_intensity_preference
            
            # Merge interaction history
            if source_user_id in self.interaction_history:
                source_interactions = self.interaction_history[source_user_id]
                if target_user_id not in self.interaction_history:
                    self.interaction_history[target_user_id] = []
                
                # Sort all interactions by timestamp and keep the most recent ones
                all_interactions = self.interaction_history[target_user_id] + source_interactions
                all_interactions.sort(key=lambda x: x.timestamp, reverse=True)
                self.interaction_history[target_user_id] = all_interactions[:self.max_interactions_per_user]
            
            # Update relationship with merged data
            self.relationships[target_user_id] = target
            
            return {
                "status": "success",
                "message": f"Merged relationship data from {source_user_id} into {target_user_id}",
                "merged_metrics": {
                    "trust": target.trust,
                    "familiarity": target.familiarity,
                    "intimacy": target.intimacy,
                    "interaction_count": target.interaction_count
                }
            }

    async def track_dominance_response(self, user_id: str, dominance_data: Dict[str, Any], user_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track user response to a dominance interaction and update relationship model accordingly.
        
        Args:
            user_id: User ID
            dominance_data: Data about the dominance interaction that occurred
            user_response: User's response to the dominance interaction
            
        Returns:
            Analysis and adaptation results
        """
        async with self._lock:
            if user_id not in self.relationships:
                return {"status": "error", "message": f"User {user_id} not found"}
                
            state = self.relationships[user_id]
            now = datetime.datetime.now()
            
            # Extract dominance interaction details
            tactic = dominance_data.get("tactic", "unknown")
            intensity = dominance_data.get("intensity", 0.5)
            context = dominance_data.get("context", "general")
            
            # Extract user response features
            response_text = user_response.get("text", "")
            response_sentiment = user_response.get("sentiment", 0.0)
            explicit_feedback = user_response.get("explicit_feedback", "none")  # "positive", "negative", "neutral"
            
            # Analyze response for compliance signals
            compliance_score = await self._analyze_compliance(response_text, user_response)
            
            # Analyze response for resistance signals
            resistance_score = await self._analyze_resistance(response_text, user_response)
            
            # Analyze response for enjoyment signals
            enjoyment_score = await self._analyze_enjoyment(response_text, user_response)
            
            # Calculate overall response quality
            response_scores = {
                "compliance": compliance_score,
                "resistance": resistance_score,
                "enjoyment": enjoyment_score,
                "sentiment": response_sentiment
            }
            
            response_quality = self._calculate_dominance_response_quality(response_scores, explicit_feedback)
            
            # Update relationship model based on response
            adaptation_results = await self._adapt_dominance_approach(
                state, 
                tactic, 
                intensity, 
                context, 
                response_quality,
                response_scores
            )
            
            # Record the interaction for history
            response_record = {
                "timestamp": now.isoformat(),
                "tactic": tactic,
                "intensity": intensity,
                "context": context,
                "response_quality": response_quality,
                "response_scores": response_scores,
                "adaptations": adaptation_results
            }
            
            # Add to dominance interaction history
            if not hasattr(state, "dominance_interaction_history"):
                state.dominance_interaction_history = []
            
            state.dominance_interaction_history.append(response_record)
            
            # Keep history to a reasonable size
            max_history = 50
            if len(state.dominance_interaction_history) > max_history:
                state.dominance_interaction_history = state.dominance_interaction_history[-max_history:]
            
            return {
                "status": "success",
                "response_quality": response_quality,
                "response_scores": response_scores,
                "adaptations": adaptation_results,
                "interaction_recorded": True
            }
    
    async def _analyze_compliance(self, text: str, response_data: Dict[str, Any]) -> float:
        """
        Analyze user response for compliance signals.
        
        Args:
            text: User response text
            response_data: Additional response data
            
        Returns:
            Compliance score (0.0-1.0)
        """
        # Basic compliance keywords
        compliance_phrases = [
            "yes", "okay", "fine", "i will", "i'll", "sure", "of course",
            "as you wish", "yes mistress", "yes master", "i obey",
            "i understand", "i'll do that", "i'll try", "i can do that"
        ]
        
        # Check for compliance signals in text
        text_lower = text.lower()
        found_phrases = [phrase for phrase in compliance_phrases if phrase in text_lower]
        phrase_score = len(found_phrases) * 0.2
        
        # Check for action compliance
        action_compliance = response_data.get("action_compliance", 0.0)
        
        # Calculate overall compliance score with bounds
        compliance_score = min(1.0, phrase_score + action_compliance)
        return compliance_score
    
    async def _analyze_resistance(self, text: str, response_data: Dict[str, Any]) -> float:
        """
        Analyze user response for resistance signals.
        
        Args:
            text: User response text
            response_data: Additional response data
            
        Returns:
            Resistance score (0.0-1.0)
        """
        # Basic resistance keywords
        resistance_phrases = [
            "no", "stop", "don't", "won't", "can't", "nope", "never",
            "i refuse", "i don't want", "too much", "that's too", "that is too",
            "i'm uncomfortable", "i am uncomfortable", "safeword", "safe word",
            "i don't like", "i hate", "i'm scared", "i am scared"
        ]
        
        # Check for resistance signals in text
        text_lower = text.lower()
        found_phrases = [phrase for phrase in resistance_phrases if phrase in text_lower]
        phrase_score = len(found_phrases) * 0.25
        
        # Check for action resistance
        action_resistance = response_data.get("action_resistance", 0.0)
        
        # Calculate overall resistance score with bounds
        resistance_score = min(1.0, phrase_score + action_resistance)
        return resistance_score
    
    async def _analyze_enjoyment(self, text: str, response_data: Dict[str, Any]) -> float:
        """
        Analyze user response for enjoyment signals.
        
        Args:
            text: User response text
            response_data: Additional response data
            
        Returns:
            Enjoyment score (0.0-1.0)
        """
        # Basic enjoyment keywords
        enjoyment_phrases = [
            "love", "enjoy", "like", "good", "great", "wonderful", "amazing",
            "perfect", "yes!", "mmm", "please", "more", "again", "thank you",
            "thank", "appreciated", "excited", "happy", "pleased"
        ]
        
        # Check for enjoyment signals in text
        text_lower = text.lower()
        found_phrases = [phrase for phrase in enjoyment_phrases if phrase in text_lower]
        phrase_score = len(found_phrases) * 0.15
        
        # Check for emotional signals
        sentiment = response_data.get("sentiment", 0.0)
        sentiment_score = max(0.0, sentiment) * 0.5  # Only count positive sentiment
        
        # Calculate overall enjoyment score with bounds
        enjoyment_score = min(1.0, phrase_score + sentiment_score)
        return enjoyment_score
    
    def _calculate_dominance_response_quality(self, scores: Dict[str, float], explicit_feedback: str) -> float:
        """
        Calculate overall quality of response to dominance interaction.
        
        Args:
            scores: Dictionary of analyzed response scores
            explicit_feedback: Any explicit feedback provided
            
        Returns:
            Response quality score (-1.0 to 1.0, where -1.0 is very negative, 1.0 is very positive)
        """
        # Give high weight to explicit feedback if available
        if explicit_feedback == "positive":
            base_quality = 0.8
        elif explicit_feedback == "negative":
            base_quality = -0.8
        else:
            # Calculate from scores
            compliance_factor = scores["compliance"] * 0.3
            resistance_factor = -scores["resistance"] * 0.4  # Negative impact
            enjoyment_factor = scores["enjoyment"] * 0.3
            
            # Sentiment has smaller weight as it's often less reliable
            sentiment_factor = scores["sentiment"] * 0.2 if "sentiment" in scores else 0.0
            
            base_quality = compliance_factor + resistance_factor + enjoyment_factor + sentiment_factor
        
        # Ensure within bounds
        return max(-1.0, min(1.0, base_quality))
    
    async def _adapt_dominance_approach(self, 
                                   state, 
                                   tactic: str, 
                                   intensity: float, 
                                   context: str,
                                   response_quality: float,
                                   response_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Adapt dominance approach based on user's response.
        
        Args:
            state: Relationship state
            tactic: Dominance tactic used
            intensity: Intensity of the dominance
            context: Context of the dominance interaction
            response_quality: Overall quality of user's response
            response_scores: Detailed response score breakdowns
            
        Returns:
            Adaptation results
        """
        adaptations = {}
        
        # Track successful and failed tactics
        if response_quality > 0.5:
            # Successfully received tactic
            if tactic not in state.successful_dominance_tactics:
                state.successful_dominance_tactics.append(tactic)
            adaptations["tactic_success"] = True
        elif response_quality < -0.3:
            # Failed tactic
            if tactic not in state.failed_dominance_tactics:
                state.failed_dominance_tactics.append(tactic)
            adaptations["tactic_failure"] = True
        
        # Adapt intensity for future interactions
        if response_quality > 0.7:
            # Very positive response - can potentially increase intensity slightly
            intensity_change = min(0.1, state.optimal_escalation_rate)
            adaptations["suggested_intensity_change"] = intensity_change
            state.current_dominance_intensity = min(1.0, state.current_dominance_intensity + intensity_change)
            adaptations["new_intensity"] = state.current_dominance_intensity
        elif response_quality < -0.5:
            # Negative response - reduce intensity
            intensity_change = -0.15
            adaptations["suggested_intensity_change"] = intensity_change
            state.current_dominance_intensity = max(0.1, state.current_dominance_intensity + intensity_change)
            adaptations["new_intensity"] = state.current_dominance_intensity
            
            # Record failed escalation if this was an escalation attempt
            if intensity > state.current_dominance_intensity:
                state.failed_escalation_attempts += 1
                adaptations["failed_escalation"] = True
        
        # If highly successful, update max achieved intensity
        if response_quality > 0.6 and intensity > state.max_achieved_intensity:
            state.max_achieved_intensity = intensity
            adaptations["new_max_intensity"] = intensity
        
        # Learn optimal escalation rate
        if "suggested_intensity_change" in adaptations and response_quality > 0.5:
            # Successful interaction - adjust optimal rate
            current_rate = state.optimal_escalation_rate
            # Slight shift toward the successful change
            rate_adjustment = (adaptations["suggested_intensity_change"] - current_rate) * 0.2
            state.optimal_escalation_rate = max(0.05, min(0.3, current_rate + rate_adjustment))
            adaptations["optimal_escalation_rate"] = state.optimal_escalation_rate
        
        # Check for limits being approached or crossed
        if response_quality < -0.3 and response_scores["resistance"] > 0.6:
            # This might be approaching a limit
            context_limit = f"{context}_{tactic}"
            if context_limit not in state.soft_limits_approached:
                state.soft_limits_approached.append(context_limit)
                adaptations["new_soft_limit_approached"] = context_limit
        
        if response_quality > 0.5 and response_scores["resistance"] > 0.3:
            # Successfully pushed through initial resistance
            context_limit = f"{context}_{tactic}"
            if context_limit in state.soft_limits_approached and context_limit not in state.soft_limits_crossed_successfully:
                state.soft_limits_crossed_successfully.append(context_limit)
                adaptations["soft_limit_crossed"] = context_limit
        
        # Hard limits - if we see very strong resistance, mark as hard limit
        if response_quality < -0.7 and response_scores["resistance"] > 0.8:
            context_limit = f"{context}_{tactic}"
            if context_limit not in state.hard_limits:
                state.hard_limits.append(context_limit)
                adaptations["new_hard_limit"] = context_limit
        
        # Update preferred style if we can determine it
        if response_quality > 0.7:
            # Check if we can infer preferred style from this highly successful interaction
            styles_map = {
                "verbal_humiliation": "verbal",
                "command": "directive",
                "physical_control": "physical",
                "psychological": "psychological",
                "teasing": "playful"
            }
            
            if tactic in styles_map:
                state.preferred_dominance_style = styles_map[tactic]
                adaptations["preferred_style"] = state.preferred_dominance_style
        
        return adaptations
    
    async def get_dominance_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        Get recommendations for dominance interactions based on learned user preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            Recommendations for dominance interactions
        """
        async with self._lock:
            if user_id not in self.relationships:
                return {"status": "error", "message": f"User {user_id} not found"}
                
            state = self.relationships[user_id]
            
            # Get current state values
            current_intensity = state.current_dominance_intensity
            max_intensity = state.max_achieved_intensity
            optimal_rate = state.optimal_escalation_rate
            preferred_style = state.preferred_dominance_style
            
            # Get successful and failed tactics
            successful_tactics = state.successful_dominance_tactics
            failed_tactics = state.failed_dominance_tactics
            
            # Get limits
            hard_limits = state.hard_limits
            soft_limits_approached = state.soft_limits_approached
            soft_limits_crossed = state.soft_limits_crossed_successfully
            
            # Calculate recommended intensity
            # Stay within bounds of what's been successful
            safe_intensity = max(0.1, min(max_intensity * 1.1, current_intensity))
            
            # Calculate potential escalation if appropriate
            escalation_intensity = min(1.0, current_intensity + optimal_rate)
            
            # Determine recommended tactics
            recommended_tactics = []
            for tactic in successful_tactics:
                # Skip tactics that hit limits
                tactic_contexts = [limit for limit in hard_limits if tactic in limit]
                if not tactic_contexts:
                    recommended_tactics.append(tactic)
            
            # Sort by most likely to succeed first
            if hasattr(state, "dominance_interaction_history"):
                tactic_success_rates = {}
                for tactic in recommended_tactics:
                    # Count successes and attempts
                    successes = 0
                    attempts = 0
                    for record in state.dominance_interaction_history:
                        if record["tactic"] == tactic:
                            attempts += 1
                            if record["response_quality"] > 0.5:
                                successes += 1
                    
                    # Calculate success rate
                    rate = successes / max(1, attempts)
                    tactic_success_rates[tactic] = rate
                
                # Sort by success rate
                recommended_tactics.sort(key=lambda t: tactic_success_rates.get(t, 0), reverse=True)
            
            # Determine if escalation is advisable
            escalation_advisable = (
                state.failed_escalation_attempts < 3 and  # Not too many failed attempts
                max_intensity > 0.3 and  # Some success with intensity
                len(successful_tactics) >= 2  # Multiple successful tactics
            )
            
            return {
                "status": "success",
                "current_intensity": current_intensity,
                "safe_intensity": safe_intensity,
                "escalation_intensity": escalation_intensity,
                "escalation_advisable": escalation_advisable,
                "recommended_tactics": recommended_tactics[:3],  # Top 3
                "preferred_style": preferred_style,
                "avoid_tactics": failed_tactics,
                "hard_limits": hard_limits,
                "soft_limits_approached": soft_limits_approached,
                "optimal_escalation_rate": optimal_rate
            }

# Create a RelationshipManagerAgent that exposes the RelationshipManager functionality
def create_relationship_manager_agent(relationship_manager: RelationshipManager) -> Agent:
    """Create an agent that provides access to the relationship manager."""
    return Agent(
        name="Relationship Manager Agent",
        instructions="""You manage Nyx's relationships with users. Your responsibilities include:
        
        1. Tracking relationship states including trust, familiarity, intimacy, and dominance
        2. Updating relationship metrics based on interactions
        3. Identifying trends and patterns in relationships
        4. Providing recommendations for relationship development
        5. Supporting dominance dynamics when appropriate
        
        Use the available tools to update and retrieve relationship information.
        """,
        tools=[
            relationship_manager.get_or_create_relationship,
            relationship_manager.get_relationship_state,
            relationship_manager.get_relationship_summary,
            relationship_manager.get_interaction_history
        ]
    )
