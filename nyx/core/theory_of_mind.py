# nyx/core/theory_of_mind.py

import logging
import datetime
import math
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

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

    def __init__(self, relationship_manager=None, multimodal_integrator=None, memory_core=None):
        self.relationship_manager = relationship_manager  # To get relationship context
        self.multimodal_integrator = multimodal_integrator  # To analyze user input modalities
        self.memory_core = memory_core  # To retrieve interaction history
        self.user_models: Dict[str, UserMentalState] = {}  # user_id -> UserMentalState
        self.inference_agent = self._create_inference_agent()
        self.update_decay_rate = 0.05  # How much confidence decays per hour without update
        self.trace_group_id = "NyxTheoryOfMind"
        self._lock = asyncio.Lock()  # For thread safety

        logger.info("TheoryOfMind initialized")

    def _create_inference_agent(self) -> Optional[Agent]:
        """Creates an agent for inferring user mental state."""
        try:
            return Agent(
                name="Mental State Inference Agent",
                instructions="""You are an expert AI psychologist analyzing interactions to infer user mental states.
                
                Given the user's input (text and modality features), Nyx's response, relationship context, and recent history, 
                infer the user's likely:
                
                1. Emotional state:
                   - Primary emotion (e.g., joy, frustration, curiosity, anxiety)
                   - Valence (-1.0 to 1.0, where -1.0 is very negative, 0.0 is neutral, 1.0 is very positive)
                   - Arousal (0.0 to 1.0, where 0.0 is calm/unenergetic, 1.0 is excited/energetic)
                
                2. Cognitive state:
                   - Current goals (what is the user trying to accomplish?)
                   - Key beliefs relevant to the interaction
                   - Focus of attention
                   - Knowledge level about the current topic (0.0 to 1.0)
                
                3. Perception of Nyx:
                   - Perceived trust (how much does the user trust Nyx?)
                   - Perceived familiarity (how familiar/comfortable is the user with Nyx?)
                
                Provide confidence levels for your inferences. Focus specifically on the *user's* state, not Nyx's.
                
                Respond ONLY with a JSON object matching the UserMentalState structure (excluding user_id and last_updated).
                """,
                model="gpt-4o",
                model_settings=ModelSettings(
                    response_format={"type": "json_object"}, 
                    temperature=0.4
                ),
                tools=[
                    self.get_emotional_markers,
                    self.get_linguistic_patterns,
                    self.get_recent_interactions
                ],
                output_type=Dict  # Expecting JSON matching UserMentalState fields
            )
        except Exception as e:
            logger.error(f"Error creating ToM inference agent: {e}")
            return None

    @function_tool
    async def get_emotional_markers(self, text: str) -> Dict[str, Any]:
        """Analyze emotional markers in text (sentiment, emotion words, etc.)"""
        # Simple markers using keyword detection, could be much more sophisticated
        emotion_keywords = {
            "joy": ["happy", "excited", "delighted", "glad", "pleased", "joy", "yay", "awesome", "love"],
            "sadness": ["sad", "unhappy", "disappointed", "upset", "down", "depressed", "grief"],
            "anger": ["angry", "mad", "annoyed", "frustrated", "irritated", "outraged"],
            "fear": ["afraid", "scared", "worried", "anxious", "terrified", "nervous"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "wow"],
            "disgust": ["disgusted", "gross", "eww", "revolting"],
            "trust": ["trust", "believe", "reliable", "honest", "faithful"],
            "anticipation": ["expect", "anticipate", "looking forward", "awaiting", "hope"]
        }
        
        # Count emotion words
        text_lower = text.lower()
        emotion_counts = {}
        for emotion, keywords in emotion_keywords.items():
            count = sum(text_lower.count(word) for word in keywords)
            if count > 0:
                emotion_counts[emotion] = count
                
        # Detect sentiment markers
        positive_markers = ["thank", "appreciate", "good", "great", "excellent", "wonderful", "love", "like"]
        negative_markers = ["no", "not", "don't", "can't", "won't", "isn't", "doesn't", "problem", "issue", "bad", "terrible"]
        
        positive_count = sum(text_lower.count(word) for word in positive_markers)
        negative_count = sum(text_lower.count(word) for word in negative_markers)
        
        # Simple sentiment score
        sentiment = 0.0
        if positive_count + negative_count > 0:
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            
        return {
            "detected_emotions": emotion_counts,
            "primary_emotion": max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral",
            "sentiment_score": sentiment,
            "positive_markers": positive_count,
            "negative_markers": negative_count,
            "has_strong_markers": bool(emotion_counts) or abs(sentiment) > 0.3
        }

    @function_tool
    async def get_linguistic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns for cognitive state indicators."""
        # Count question marks as indicators of inquiry/curiosity
        question_count = text.count("?")
        
        # Look for command structures
        command_indicators = ["please", "could you", "can you", "would you", "help me", "tell me", "show me", "give me"]
        command_count = sum(text.lower().count(indicator) for indicator in command_indicators)
        
        # Check for uncertainty markers
        uncertainty_markers = ["maybe", "perhaps", "not sure", "possibly", "might", "could be", "think", "guess"]
        uncertainty_count = sum(text.lower().count(marker) for marker in uncertainty_markers)
        
        # Check for first-person pronouns (self-focus)
        self_focus_markers = ["i", "me", "my", "mine", "myself"]
        self_focus_count = sum(text.lower().count(f" {marker} ") for marker in self_focus_markers)
        
        # Check for second-person pronouns (Nyx-focus)
        nyx_focus_markers = ["you", "your", "yours", "yourself"]
        nyx_focus_count = sum(text.lower().count(f" {marker} ") for marker in nyx_focus_markers)
        
        # Calculate cognitive focus
        total_markers = self_focus_count + nyx_focus_count
        self_focus_ratio = self_focus_count / max(1, total_markers)
        
        # Analyze sentence structure
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))
        
        patterns = {
            "inquiry": question_count > 0,
            "directive": command_count > 0,
            "uncertainty": uncertainty_count > 0,
            "self_focus_ratio": self_focus_ratio,
            "avg_sentence_length": avg_sentence_length,
            "sentence_count": len(sentences),
            "short_responses": avg_sentence_length < 15 and len(sentences) <= 2
        }
        
        # Infer potential cognitive states
        cognitive_states = []
        if question_count > 1:
            cognitive_states.append("information seeking")
        if command_count > 1:
            cognitive_states.append("task focused")
        if uncertainty_count > 1:
            cognitive_states.append("uncertain/exploring")
        if self_focus_ratio > 0.7:
            cognitive_states.append("self-focused")
        elif self_focus_ratio < 0.3 and nyx_focus_count > 0:
            cognitive_states.append("Nyx-focused")
            
        patterns["inferred_cognitive_states"] = cognitive_states
        
        return patterns

    @function_tool
    async def get_recent_interactions(self, user_id: str, limit: int = 3) -> Dict[str, Any]:
        """Retrieve recent interaction history for context."""
        interactions = []
        
        if self.memory_core:
            try:
                # Get recent interaction memories
                memories = await self.memory_core.retrieve_memories(
                    query=f"recent interactions with {user_id}",
                    limit=limit,
                    memory_types=["interaction", "experience"]
                )
                
                for memory in memories:
                    interactions.append({
                        "timestamp": memory.get("timestamp"),
                        "content": memory.get("content", "")[:200],  # Truncate long content
                        "valence": memory.get("valence", 0),
                        "tags": memory.get("tags", [])
                    })
            except Exception as e:
                logger.error(f"Error retrieving memories: {e}")
        
        # Get relationship info if available
        relationship_data = {}
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if relationship:
                    relationship_data = {
                        "trust": relationship.trust,
                        "familiarity": relationship.familiarity,
                        "intimacy": relationship.intimacy,
                        "interaction_count": relationship.interaction_count
                    }
            except Exception as e:
                logger.error(f"Error retrieving relationship data: {e}")
        
        return {
            "recent_interactions": interactions,
            "relationship_data": relationship_data,
            "has_history": len(interactions) > 0
        }

    def _get_or_create_user_model(self, user_id: str) -> UserMentalState:
        """Gets or creates the mental state model for a user."""
        if user_id not in self.user_models:
            logger.info(f"Creating new mental state model for user '{user_id}'")
            self.user_models[user_id] = UserMentalState(user_id=user_id)
        return self.user_models[user_id]

    async def update_user_model(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the inferred mental state model for a user based on the latest interaction.

        Args:
            user_id: The ID of the user.
            interaction_data: Dictionary containing details of the interaction, e.g.,
                'user_input': Raw user input (text or processed percept).
                'nyx_response': Nyx's response text.
                'emotional_context': Nyx's emotional state during interaction.
                'modality_features': Extracted features from multimodal input.
                'recent_history': List of recent turns.
        """
        if not self.inference_agent:
            logger.warning("Cannot update user model: Inference agent not available")
            return {"status": "error", "reason": "Inference agent not available"}

        async with self._lock:
            state = self._get_or_create_user_model(user_id)
            now = datetime.datetime.now()

            # Apply confidence decay before update
            elapsed_hours = (now - state.last_updated).total_seconds() / 3600.0
            decay_factor = math.exp(-self.update_decay_rate * elapsed_hours)
            state.overall_confidence *= decay_factor
            state.emotion_confidence *= decay_factor

        logger.debug(f"Inferring mental state for user '{user_id}'")
        try:
            with trace(workflow_name="InferUserMentalState", group_id=self.trace_group_id, metadata={"user_id": user_id}):
                # Prepare context for the agent
                user_input = interaction_data.get("user_input", "")
                
                # Get modality information if available
                modality_info = {}
                if self.multimodal_integrator and "modality_features" in interaction_data:
                    # Assuming modality_features contains extracted features from various modalities
                    modality_info = interaction_data["modality_features"]
                
                # Prepare the full context for inference
                context = {
                    "user_id": user_id,
                    "current_interaction": {
                        "user_input": user_input,
                        "nyx_response": interaction_data.get("nyx_response", ""),
                        "modality_features": modality_info
                    },
                    "previous_state": state.model_dump(exclude={'user_id', 'last_updated'})
                }
                
                # Run the inference agent
                result = await Runner.run(
                    self.inference_agent,
                    json.dumps(context),
                    run_config={
                        "workflow_name": "MentalStateInference",
                        "trace_metadata": {"user_id": user_id}
                    }
                )
                
                # Extract inferred data
                inferred_data = result.final_output
                
                # Update the user model with weighted blending
                await self._update_model_with_inference(state, inferred_data)
                
                # Log update summary
                logger.info(f"Updated mental state model for user '{user_id}': Emotion={state.inferred_emotion} (Conf: {state.emotion_confidence:.2f})")
                
                # Return the updated state
                return {
                    "status": "success",
                    "user_id": user_id,
                    "inferred_emotion": state.inferred_emotion,
                    "emotion_confidence": state.emotion_confidence,
                    "valence": state.valence,
                    "arousal": state.arousal,
                    "overall_confidence": state.overall_confidence,
                    "inferred_goals": state.inferred_goals[:3]  # Return top 3 goals only
                }

        except Exception as e:
            logger.exception(f"Error updating user model for '{user_id}': {e}")
            
            async with self._lock:
                # Slightly decay confidence on error
                state.overall_confidence *= 0.9
                state.last_updated = now
                
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e)
            }

    async def _update_model_with_inference(self, state: UserMentalState, inferred_data: Dict[str, Any]) -> None:
        """Update the user model with weighted blending from new inference."""
        async with self._lock:
            # Calculate blend weights based on confidence
            inference_confidence = inferred_data.get("overall_confidence", 0.5)
            update_weight = inference_confidence * 0.5  # How much the new inference influences the state
            current_weight = 1.0 - update_weight
            
            # Update emotional state
            if "inferred_emotion" in inferred_data:
                state.inferred_emotion = inferred_data["inferred_emotion"]
            
            if "emotion_confidence" in inferred_data:
                state.emotion_confidence = state.emotion_confidence * current_weight + inferred_data["emotion_confidence"] * update_weight
            
            if "valence" in inferred_data:
                state.valence = state.valence * current_weight + inferred_data["valence"] * update_weight
            
            if "arousal" in inferred_data:
                state.arousal = state.arousal * current_weight + inferred_data["arousal"] * update_weight
            
            # Update cognitive state
            if "inferred_goals" in inferred_data:
                # Combine goals with priority to new ones
                existing_goals = set(state.inferred_goals)
                new_goals = inferred_data["inferred_goals"]
                
                # Add new goals at the beginning
                combined_goals = []
                for goal in new_goals:
                    if goal not in existing_goals:
                        combined_goals.append(goal)
                
                # Add existing goals that aren't in new goals
                for goal in state.inferred_goals:
                    if goal not in new_goals:
                        combined_goals.append(goal)
                
                # Update with combined list, limited to reasonable size
                state.inferred_goals = combined_goals[:5]  # Keep at most 5 goals
            
            if "inferred_beliefs" in inferred_data:
                # Update beliefs, keeping existing ones if not contradicted
                for belief, value in inferred_data["inferred_beliefs"].items():
                    state.inferred_beliefs[belief] = value
            
            if "attention_focus" in inferred_data:
                state.attention_focus = inferred_data["attention_focus"]
            
            if "knowledge_level" in inferred_data:
                state.knowledge_level = state.knowledge_level * current_weight + inferred_data["knowledge_level"] * update_weight
            
            # Update relationship perspective
            if "perceived_trust" in inferred_data:
                state.perceived_trust = state.perceived_trust * current_weight + inferred_data["perceived_trust"] * update_weight
            
            if "perceived_familiarity" in inferred_data:
                state.perceived_familiarity = state.perceived_familiarity * current_weight + inferred_data["perceived_familiarity"] * update_weight
            
            # Update overall confidence
            if "overall_confidence" in inferred_data:
                state.overall_confidence = state.overall_confidence * current_weight + inference_confidence * update_weight
            
            # Update timestamp
            state.last_updated = datetime.datetime.now()

    async def get_user_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Gets the current inferred mental state for a user."""
        async with self._lock:
            if user_id not in self.user_models:
                return None
                
            state = self.user_models[user_id]
            
            # Apply decay before returning if it's been a while
            now = datetime.datetime.now()
            elapsed_hours = (now - state.last_updated).total_seconds() / 3600.0
            if elapsed_hours > 0.1:  # More than ~6 mins
                decay_factor = math.exp(-self.update_decay_rate * elapsed_hours)
                state.overall_confidence *= decay_factor
                state.emotion_confidence *= decay_factor
                state.last_updated = now  # Update timestamp to prevent rapid decay on multiple gets
                
            return state.model_dump()
            
    async def reset_user_model(self, user_id: str) -> Dict[str, Any]:
        """Resets the mental model for a user to baseline."""
        async with self._lock:
            if user_id in self.user_models:
                old_state = self.user_models[user_id].model_dump()
                self.user_models[user_id] = UserMentalState(user_id=user_id)
                logger.info(f"Reset mental state model for user '{user_id}'")
                return {
                    "status": "success",
                    "message": f"Mental state model for user '{user_id}' has been reset",
                    "previous_state": old_state
                }
            else:
                return {
                    "status": "error",
                    "message": f"No mental state model found for user '{user_id}'"
                }
                
    async def merge_user_models(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Merges mental models from source to target user (for user ID reconciliation)."""
        async with self._lock:
            if source_id not in self.user_models:
                return {
                    "status": "error",
                    "message": f"Source user '{source_id}' not found"
                }
                
            if target_id not in self.user_models:
                # If target doesn't exist, simply rename the source
                self.user_models[target_id] = self.user_models[source_id]
                del self.user_models[source_id]
                return {
                    "status": "success",
                    "message": f"Moved model from '{source_id}' to '{target_id}'"
                }
            
            # Both exist, need to blend them
            source = self.user_models[source_id]
            target = self.user_models[target_id]
            
            # Determine blend weights based on confidence and recency
            source_weight = source.overall_confidence * 0.5
            source_recency = (datetime.datetime.now() - source.last_updated).total_seconds()
            target_recency = (datetime.datetime.now() - target.last_updated).total_seconds()
            
            # More recent model gets higher weight
            if source_recency < target_recency:
                source_weight += 0.2
            
            target_weight = 1.0 - source_weight
            
            # Blend numerical values
            target.valence = source.valence * source_weight + target.valence * target_weight
            target.arousal = source.arousal * source_weight + target.arousal * target_weight
            target.knowledge_level = source.knowledge_level * source_weight + target.knowledge_level * target_weight
            target.perceived_trust = source.perceived_trust * source_weight + target.perceived_trust * target_weight
            target.perceived_familiarity = source.perceived_familiarity * source_weight + target.perceived_familiarity * target_weight
            
            # Take the more confident emotional state
            if source.emotion_confidence > target.emotion_confidence:
                target.inferred_emotion = source.inferred_emotion
                target.emotion_confidence = source.emotion_confidence
            
            # Combine goals and beliefs
            combined_goals = list(set(target.inferred_goals + source.inferred_goals))
            target.inferred_goals = combined_goals[:5]  # Keep top 5 only
            
            # Merge beliefs, keeping more confident ones
            for belief, value in source.inferred_beliefs.items():
                if belief not in target.inferred_beliefs:
                    target.inferred_beliefs[belief] = value
            
            # Take the more recent attention focus
            if (source.last_updated > target.last_updated and source.attention_focus):
                target.attention_focus = source.attention_focus
            
            # Set confidence to max of the two
            target.overall_confidence = max(target.overall_confidence, source.overall_confidence)
            
            # Update and return
            target.last_updated = datetime.datetime.now()
            self.user_models[target_id] = target
            
            # Optionally remove source
            del self.user_models[source_id]
            
            return {
                "status": "success",
                "message": f"Merged mental state models from '{source_id}' to '{target_id}'",
                "merged_model": target.model_dump()
            }
@function_tool
async def detect_humiliation_signals(self, text: str) -> Dict[str, Any]:
    """Detect signals of humiliation, embarrassment, or discomfort in user text."""
    humiliation_markers = [
        "embarrassed", "humiliated", "ashamed", "blushing", "awkward",
        "uncomfortable", "exposed", "vulnerable", "pathetic", "inadequate",
        "sorry", "please don't", "stop laughing", "don't laugh", "begging"
    ]
    
    # Check for humiliation markers
    marker_count = sum(text.lower().count(marker) for marker in humiliation_markers)
    
    # Check for self-deprecation
    self_deprecation_markers = ["i'm bad", "i failed", "i can't", "i'm not good enough", "i'm pathetic"]
    self_deprecation_count = sum(text.lower().count(marker) for marker in self_deprecation_markers)
    
    return {
        "humiliation_detected": marker_count > 0 or self_deprecation_count > 0,
        "intensity": min(1.0, (marker_count * 0.2) + (self_deprecation_count * 0.3)),
        "marker_count": marker_count,
        "self_deprecation_count": self_deprecation_count
    }
