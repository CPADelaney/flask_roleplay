# nyx/core/relationship_manager.py

import logging
import datetime
import math
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

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
    key_memories: List[str] = Field(default_factory=list, description="IDs of key memories shared/related to this user") # Max ~10-20?
    inferred_user_traits: Dict[str, float] = Field(default_factory=dict, description="Nyx's inferred traits about the user")
    shared_secrets_level: float = Field(0.0, ge=0.0, le=1.0) # How much sensitive info shared
    current_dominance_intensity: float = Field(0.0, ge=0.0, le=1.0, description="Current level of expressed dominance in the interaction")
    max_achieved_intensity: float = Field(0.0, ge=0.0, le=1.0, description="Highest intensity level successfully reached with this user")
    failed_escalation_attempts: int = 0

class RelationshipManager:
    """Manages Nyx's relationship states with different users."""

    def __init__(self, memory_orchestrator=None):
        self.memory_orchestrator = memory_orchestrator # For storing/retrieving relationship-linked memories
        self.relationships: Dict[str, RelationshipState] = {} # user_id -> RelationshipState
        self.update_weights = {
            "positive_valence": 0.05, # How much positive emotion increases trust/intimacy
            "negative_valence": 0.08, # How much negative emotion increases conflict/decreases trust
            "shared_experience": 0.03, # Impact of sharing an experience
            "shared_reflection": 0.05, # Impact of sharing a reflection
            "goal_success_user": 0.04, # Impact of success on user-related goal
            "goal_failure_user": 0.06, # Impact of failure on user-related goal
            "user_feedback_positive": 0.1,
            "user_feedback_negative": 0.15,
            "time_decay_factor": 0.998, # Slow decay per day for trust/intimacy if no interaction
        }
        logger.info("RelationshipManager initialized.")

    def _get_or_create_relationship(self, user_id: str) -> RelationshipState:
        """Gets the relationship state for a user, creating it if it doesn't exist."""
        if user_id not in self.relationships:
            logger.info(f"Creating new relationship profile for user '{user_id}'.")
            self.relationships[user_id] = RelationshipState(user_id=user_id)
        return self.relationships[user_id]

    async def update_relationship_on_interaction(
        self,
        user_id: str,
        interaction_data: Dict[str, Any] # e.g., user_input, nyx_response, emotional_context, feedback, goal_outcome
    ):
        """Updates the relationship state based on a completed interaction."""
        if not user_id: return

        state = self._get_or_create_relationship(user_id)
        now = datetime.datetime.now()

        # Decay factors since last interaction
        time_decay = 1.0
        if state.last_interaction_time:
             days_since = (now - state.last_interaction_time).total_seconds() / (3600 * 24)
             time_decay = math.pow(self.update_weights["time_decay_factor"], days_since)
             state.trust *= time_decay
             state.intimacy *= time_decay
             state.familiarity *= time_decay # Familiarity also decays slightly

        state.interaction_count += 1
        state.last_interaction_time = now

        # Update based on emotional context of interaction
        emo_context = interaction_data.get("emotional_context", {})
        valence = emo_context.get("valence", 0.0)
        primary_emotion = emo_context.get("primary_emotion", {}).get("name")

        if valence > 0.3: # Positive interaction
            state.positive_interaction_score += valence
            state.trust = min(1.0, state.trust + self.update_weights["positive_valence"] * valence)
            state.familiarity = min(1.0, state.familiarity + 0.01) # Familiarity increases slowly
            if primary_emotion in ["Joy", "Trust", "Love"]:
                 state.intimacy = min(1.0, state.intimacy + 0.02 * valence)
        elif valence < -0.3: # Negative interaction
             state.negative_interaction_score += abs(valence)
             state.trust = max(0.0, state.trust - self.update_weights["negative_valence"] * abs(valence))
             state.conflict = min(1.0, state.conflict + 0.05 * abs(valence)) # Conflict increases
             if primary_emotion in ["Anger", "Disgust", "Fear"]:
                  state.intimacy = max(0.0, state.intimacy - 0.03 * abs(valence)) # Reduces intimacy

        # Update based on experience sharing
        if interaction_data.get("shared_experience"):
             state.familiarity = min(1.0, state.familiarity + 0.02)
             state.intimacy = min(1.0, state.intimacy + self.update_weights["shared_experience"])
        if interaction_data.get("shared_reflection"):
             state.familiarity = min(1.0, state.familiarity + 0.03)
             state.intimacy = min(1.0, state.intimacy + self.update_weights["shared_reflection"])

        # Update based on user feedback
        feedback = interaction_data.get("user_feedback") # Expects e.g., {"rating": 0.8, "type": "positive"}
        if feedback:
             rating = feedback.get("rating", 0.5)
             if feedback.get("type") == "positive":
                  state.trust = min(1.0, state.trust + self.update_weights["user_feedback_positive"] * rating)
                  state.positive_interaction_score += rating
             elif feedback.get("type") == "negative":
                  state.trust = max(0.0, state.trust - self.update_weights["user_feedback_negative"] * rating)
                  state.negative_interaction_score += rating
                  state.conflict = min(1.0, state.conflict + 0.08 * rating)

        # Update based on goal outcomes related to user
        goal_outcome = interaction_data.get("goal_outcome") # Expects e.g., {"status": "completed", "priority": 0.8}
        if goal_outcome:
             status = goal_outcome.get("status")
             priority = goal_outcome.get("priority", 0.5)
             if status == "completed":
                  state.trust = min(1.0, state.trust + self.update_weights["goal_success_user"] * priority)
                  state.familiarity = min(1.0, state.familiarity + 0.01 * priority)
             elif status == "failed":
                  state.trust = max(0.0, state.trust - self.update_weights["goal_failure_user"] * priority)
                  state.conflict = min(1.0, state.conflict + 0.04 * priority)

        # Update dominance balance (simplified - based on interaction style tags?)
        nyx_style = interaction_data.get("nyx_response_style", "")
        user_style = interaction_data.get("user_input_style", "")
        if "dominant" in nyx_style and "submissive" in user_style: state.dominance_balance = min(1.0, state.dominance_balance + 0.05)
        elif "submissive" in nyx_style and "dominant" in user_style: state.dominance_balance = max(-1.0, state.dominance_balance - 0.05)
        elif "dominant" in nyx_style and "dominant" in user_style: state.conflict = min(1.0, state.conflict + 0.02) # Dominance clash -> conflict
        else: state.dominance_balance *= 0.98 # Drift towards neutral

        # Link key memories (e.g., high significance, high emotion, shared secrets)
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

        logger.debug(f"Updated relationship for user '{user_id}': Trust={state.trust:.2f}, Familiarity={state.familiarity:.2f}, Intimacy={state.intimacy:.2f}, Conflict={state.conflict:.2f}")

    async def get_relationship_state(self, user_id: str) -> Optional[RelationshipState]:
        """Gets the current relationship state for a user."""
        # Apply time decay before returning
        state = self._get_or_create_relationship(user_id)
        now = datetime.datetime.now()
        if state.last_interaction_time:
             days_since = (now - state.last_interaction_time).total_seconds() / (3600 * 24)
             if days_since > 0.5: # Only apply decay if significant time passed
                  time_decay = math.pow(self.update_weights["time_decay_factor"], days_since)
                  state.trust *= time_decay
                  state.intimacy *= time_decay
                  state.familiarity *= time_decay
                  # Clamp again after decay
                  state.trust = max(0.0, min(1.0, state.trust))
                  state.familiarity = max(0.0, min(1.0, state.familiarity))
                  state.intimacy = max(0.0, min(1.0, state.intimacy))
                  # Don't decay conflict or dominance balance this way
                  state.last_interaction_time = now # Update time to prevent repeated decay on consecutive gets
        return state

    async def get_relationship_summary(self, user_id: str) -> str:
         """Provides a brief textual summary of the relationship."""
         state = await self.get_relationship_state(user_id)
         if not state: return "No relationship data available."

         summary = f"Relationship with {user_id}: "
         if state.trust > 0.8: summary += "Very High Trust. "
         elif state.trust > 0.6: summary += "High Trust. "
         elif state.trust > 0.4: summary += "Moderate Trust. "
         else: summary += "Low Trust. "

         if state.familiarity > 0.7: summary += "Very Familiar. "
         elif state.familiarity > 0.4: summary += "Familiar. "
         else: summary += "Getting Acquainted. "

         if state.intimacy > 0.6: summary += "High Intimacy. "
         elif state.intimacy > 0.3: summary += "Moderate Intimacy. "
         else: summary += "Low Intimacy. "

         if state.conflict > 0.5: summary += f"Notable Conflict (Level: {state.conflict:.2f}). "

         dom = state.dominance_balance
         if dom > 0.4: summary += "Nyx is dominant. "
         elif dom < -0.4: summary += "User is dominant. "
         else: summary += "Balanced dynamic. "

         return summary.strip()
