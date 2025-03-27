# nyx/core/theory_of_mind.py

import logging
import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import asyncio

# Assume Agent SDK is available
try:
    from agents import Agent, Runner, ModelSettings, trace
    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False
    # Dummy classes if SDK not found
    class Agent: pass
    class Runner: pass
    class ModelSettings: pass
    def trace(workflow_name, group_id):
         # ... (dummy trace context manager) ...
         pass

logger = logging.getLogger(__name__)

class UserMentalState(BaseModel):
    """Represents the inferred mental state of a user."""
    user_id: str
    # Emotional State
    inferred_emotion: str = Field("neutral", description="Most likely current emotion")
    emotion_confidence: float = Field(0.5, ge=0.0, le=1.0)
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    # Cognitive State
    inferred_goals: List[str] = Field(default_factory=list, description="User's likely immediate goals")
    inferred_beliefs: Dict[str, Any] = Field(default_factory=dict, description="Key beliefs relevant to the interaction")
    attention_focus: Optional[str] = None
    knowledge_level: float = Field(0.5, ge=0.0, le=1.0, description="Estimated knowledge about the current topic")
    # Relationship Perspective
    perceived_trust: float = Field(0.5, ge=0.0, le=1.0, description="User's perceived trust in Nyx")
    perceived_familiarity: float = Field(0.1, ge=0.0, le=1.0)
    # Confidence
    overall_confidence: float = Field(0.4, ge=0.0, le=1.0, description="Overall confidence in the inferred state")
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)

class TheoryOfMind:
    """
    Models the mental states of interaction partners (primarily the user).
    Infers beliefs, goals, and emotions based on interaction history.
    """

    def __init__(self, relationship_manager=None, multimodal_integrator=None):
        self.relationship_manager = relationship_manager # To get relationship context
        self.multimodal_integrator = multimodal_integrator # To analyze user input modalities
        self.user_models: Dict[str, UserMentalState] = {} # user_id -> UserMentalState
        self.inference_agent = self._create_inference_agent()
        self.update_decay_rate = 0.05 # How much confidence decays per hour without update
        self.trace_group_id = "NyxTheoryOfMind"

        logger.info("TheoryOfMind initialized.")

    def _create_inference_agent(self) -> Optional[Agent]:
        """Creates an agent for inferring user mental state."""
        if not AGENT_SDK_AVAILABLE: return None
        try:
            return Agent(
                name="User State Inference Agent",
                instructions="""You are an expert AI psychologist analyzing interactions to infer a user's mental state (emotions, goals, beliefs).
                Given the user's input (text, potentially modality features), Nyx's response, the current relationship state, and recent history, infer the user's likely emotional state (emotion, valence, arousal), immediate goals, and key beliefs relevant to the interaction.
                Also estimate the user's perception of trust and familiarity towards Nyx.
                Provide confidence levels for your inferences. Focus on the *user's* state, not Nyx's.
                Respond ONLY with a JSON object matching the UserMentalState structure (excluding user_id and last_updated).
                """,
                model="gpt-4o", # Needs strong inference capabilities
                model_settings=ModelSettings(response_format={"type": "json_object"}, temperature=0.4),
                # Tools could include sentiment analysis, topic modeling etc. if needed
                output_type=Dict # Expecting JSON matching UserMentalState fields
            )
        except Exception as e:
            logger.error(f"Error creating ToM inference agent: {e}")
            return None

    def _get_or_create_user_model(self, user_id: str) -> UserMentalState:
        """Gets or creates the mental state model for a user."""
        if user_id not in self.user_models:
            logger.info(f"Creating new mental state model for user '{user_id}'.")
            self.user_models[user_id] = UserMentalState(user_id=user_id)
        return self.user_models[user_id]

    async def update_user_model(self, user_id: str, interaction_data: Dict[str, Any]):
        """
        Updates the inferred mental state model for a user based on the latest interaction.

        Args:
            user_id: The ID of the user.
            interaction_data: Dictionary containing details of the interaction, e.g.,
                'user_input': Raw user input (text or processed percept).
                'nyx_response': Nyx's response text.
                'emotional_context': Nyx's emotional state during interaction.
                'relationship_state': Current RelationshipState object.
                'recent_history': List of recent turns.
        """
        if not self.inference_agent:
            logger.warning("Cannot update user model: Inference agent not available.")
            return

        state = self._get_or_create_user_model(user_id)
        now = datetime.datetime.now()

        # Apply confidence decay before update
        elapsed_hours = (now - state.last_updated).total_seconds() / 3600.0
        decay_factor = math.exp(-self.update_decay_rate * elapsed_hours)
        state.overall_confidence *= decay_factor
        state.emotion_confidence *= decay_factor

        logger.debug(f"Inferring user '{user_id}' mental state...")
        try:
            with trace(workflow_name="InferUserMentalState", group_id=self.trace_group_id, metadata={"user_id": user_id}):
                # Prepare prompt/context for the agent
                prompt_context = {
                    "user_input": interaction_data.get("user_input"),
                    "nyx_response": interaction_data.get("nyx_response"),
                    "relationship_summary": await self.relationship_manager.get_relationship_summary(user_id) if self.relationship_manager else "N/A",
                    "recent_history_summary": [f"{turn.get('role', '?')}: {str(turn.get('content', ''))[:50]}..." for turn in interaction_data.get("recent_history", [])[-4:]],
                    "previous_inferred_state": state.model_dump(exclude={'user_id', 'last_updated'}) # Pass previous state
                }
                prompt = f"Infer user mental state based on this interaction context:\n{json.dumps(prompt_context, indent=2)}"

                result = await Runner.run(self.inference_agent, prompt)
                inferred_data = json.loads(result.final_output)

                # --- Update the UserMentalState model ---
                # Use weighted update based on inference confidence
                inference_confidence = inferred_data.get("overall_confidence", 0.5)
                update_weight = inference_confidence * 0.5 # How much the new inference influences the state
                current_weight = 1.0 - update_weight

                state.inferred_emotion = inferred_data.get("inferred_emotion", state.inferred_emotion)
                state.emotion_confidence = state.emotion_confidence * current_weight + inferred_data.get("emotion_confidence", 0.5) * update_weight
                state.valence = state.valence * current_weight + inferred_data.get("valence", 0.0) * update_weight
                state.arousal = state.arousal * current_weight + inferred_data.get("arousal", 0.5) * update_weight

                # Update goals - maybe append new goals, requires more complex logic
                state.inferred_goals = inferred_data.get("inferred_goals", state.inferred_goals)
                # Update beliefs - merge dictionaries?
                state.inferred_beliefs.update(inferred_data.get("inferred_beliefs", {}))
                state.attention_focus = inferred_data.get("attention_focus", state.attention_focus)
                state.knowledge_level = state.knowledge_level * current_weight + inferred_data.get("knowledge_level", 0.5) * update_weight

                state.perceived_trust = state.perceived_trust * current_weight + inferred_data.get("perceived_trust", 0.5) * update_weight
                state.perceived_familiarity = state.perceived_familiarity * current_weight + inferred_data.get("perceived_familiarity", 0.1) * update_weight

                # Update overall confidence
                state.overall_confidence = state.overall_confidence * current_weight + inference_confidence * update_weight

                state.last_updated = now
                logger.info(f"Updated mental state model for user '{user_id}'. Inferred emotion: {state.inferred_emotion} (Conf: {state.emotion_confidence:.2f})")

        except Exception as e:
            logger.exception(f"Error updating user model for '{user_id}': {e}")
            # Slightly decay confidence on error
            state.overall_confidence *= 0.9
            state.last_updated = now


    def get_user_model(self, user_id: str) -> Optional[UserMentalState]:
        """Gets the current inferred mental state for a user."""
        if user_id in self.user_models:
             state = self.user_models[user_id]
             # Apply decay before returning if it's been a while
             now = datetime.datetime.now()
             elapsed_hours = (now - state.last_updated).total_seconds() / 3600.0
             if elapsed_hours > 0.1: # More than ~6 mins
                  decay_factor = math.exp(-self.update_decay_rate * elapsed_hours)
                  state.overall_confidence *= decay_factor
                  state.emotion_confidence *= decay_factor
                  state.last_updated = now # Update timestamp to prevent rapid decay on multiple gets
             return state
        return None
