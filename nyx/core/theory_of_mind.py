# nyx/core/theory_of_mind.py

import logging
import datetime
import math
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper, handoff, GuardrailFunctionOutput, InputGuardrail

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

class SubmissionMarkers(BaseModel):
    """Detection results for submission signals in user text."""
    overall_submission: float = Field(0.0, ge=0.0, le=1.0, description="Overall submission level detected (0-1)")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in the assessment")
    submission_type: str = Field("none", description="Primary type of submission detected")
    detected_markers: Dict[str, int] = Field(default_factory=dict, description="Count of markers by category")
    subspace_indicators: Dict[str, float] = Field(default_factory=dict, description="Indicators of subspace")
    resistance_indicators: Dict[str, float] = Field(default_factory=dict, description="Indicators of resistance")
    compliance_indicators: Dict[str, float] = Field(default_factory=dict, description="Indicators of compliance")

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
                model="gpt-4.1-nano",
                model_settings=ModelSettings(
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
                        "trust": relationship.get("trust", 0.5),
                        "familiarity": relationship.get("familiarity", 0.1),
                        "intimacy": relationship.get("intimacy", 0.1),
                        "interaction_count": relationship.get("interaction_count", 0)
                    }
            except Exception as e:
                logger.error(f"Error retrieving relationship data: {e}")
        
        return {
            "recent_interactions": interactions,
            "relationship_data": relationship_data,
            "has_history": len(interactions) > 0
        }

    @function_tool
    async def _get_or_create_user_model(self, user_id: str) -> Dict[str, Any]:
        """Gets or creates the mental state model for a user."""
        if user_id not in self.user_models:
            logger.info(f"Creating new mental state model for user '{user_id}'")
            self.user_models[user_id] = UserMentalState(user_id=user_id)
        return self.user_models[user_id].model_dump()

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
            state_dict = await self._get_or_create_user_model(user_id)
            state = UserMentalState(**state_dict)
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
            
            # Update the model in the dictionary 
            self.user_models[state.user_id] = state

    @function_tool
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
            
    @function_tool
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

    @function_tool
    async def detect_submission_signals(self, text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Detects signals of submission, compliance, resistance, and subspace in user text.
        
        Args:
            text: User's message text
            user_id: Optional user ID for contextual analysis
            
        Returns:
            Detected submission signals
        """
        # Initialize result
        result = {"overall_submission": 0.0, "confidence": 0.0, "submission_type": "none"}
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Define marker categories and patterns
        submission_markers = {
            "deference": [
                "yes mistress", "yes goddess", "yes ma'am", "as you wish", 
                "as you command", "whatever you want", "i obey", "i'll obey",
                "at your service", "your wish", "your command", "your pleasure",
                "i submit", "i surrender", "i yield"
            ],
            "begging": [
                "please", "i beg", "begging", "desperate", "mercy",
                "i need", "allow me", "let me", "may i", "permission"
            ],
            "honorifics": [
                "mistress", "goddess", "ma'am", "miss", "my queen", "my lady",
                "superior", "owner", "controller"
            ],
            "self_diminishment": [
                "this slave", "your slave", "this servant", "your servant",
                "this pet", "your pet", "this toy", "your toy", "worthless",
                "pathetic", "weak", "inferior", "unworthy"
            ],
            "vulnerability": [
                "exposed", "vulnerable", "helpless", "weak", "powerless",
                "at your mercy", "dependent", "reliant", "needs you"
            ]
        }
        
        # Resistance markers
        resistance_markers = {
            "direct_refusal": [
                "no", "won't", "can't", "refuse", "not going to", 
                "will not", "don't want to", "won't let you", "stop"
            ],
            "bargaining": [
                "instead", "rather", "alternative", "compromise", "negotiation",
                "another way", "different approach", "not that"
            ],
            "questioning": [
                "why should", "why would", "what gives you", "who says", 
                "what makes you think", "why do you think",
                "what right", "since when"
            ],
            "assertion": [
                "i decide", "my choice", "my decision", "my right", 
                "i am in control", "i choose", "i will determine", "i don't have to"
            ]
        }
        
        # Compliance markers
        compliance_markers = {
            "agreement": [
                "i will", "i'll", "yes", "okay", "fine", "sure",
                "alright", "as you say", "understood", "noted"
            ],
            "action_confirmation": [
                "i did", "i have", "completed", "done", "finished",
                "carried out", "performed", "executed", "accomplished"
            ],
            "eager_service": [
                "happy to", "glad to", "eager to", "excited to",
                "looking forward to", "delighted to", "can't wait to"
            ]
        }
        
        # Subspace indicators
        subspace_indicators = {
            "simplification": [
                "simple sentences", "brief responses", "one-word answers",
                "minimal text", "reduced complexity"
            ],
            "disorientation": [
                "confused", "disoriented", "foggy", "hazy", "floating",
                "drifting", "detached", "distant"
            ],
            "heightened_emotion": [
                "overwhelmed", "intense", "deep", "profound", "consuming",
                "powerful", "strong feelings", "emotional"
            ],
            "repetition": [
                "repeated phrases", "echoing", "same words", "repeating"
            ]
        }
        
        # Count markers in each category
        detected_markers = {}
        
        for category, patterns in submission_markers.items():
            count = sum(text_lower.count(pattern) for pattern in patterns)
            if count > 0:
                detected_markers[category] = count
        
        # Check resistance markers
        resistance_counts = {}
        for category, patterns in resistance_markers.items():
            count = sum(text_lower.count(pattern) for pattern in patterns)
            if count > 0:
                resistance_counts[category] = count
        
        # Check compliance markers
        compliance_counts = {}
        for category, patterns in compliance_markers.items():
            count = sum(text_lower.count(pattern) for pattern in patterns)
            if count > 0:
                compliance_counts[category] = count
        
        # Advanced linguistic pattern analysis for subspace
        subspace_scores = {}
        
        # 1. Check for simplified language/grammar
        words = text_lower.split()
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        sentences = [s.strip() for s in text_lower.split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # Simpler language can indicate subspace
        if avg_word_length < 4.0:
            subspace_scores["simplified_vocabulary"] = min(1.0, (4.0 - avg_word_length) / 2.0)
        
        if avg_sentence_length < 5.0:
            subspace_scores["simplified_grammar"] = min(1.0, (5.0 - avg_sentence_length) / 3.0)
        
        # 2. Check for repetition (can indicate trance/subspace)
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repetition_ratio = sum(1 for count in word_counts.values() if count > 1) / max(1, len(word_counts))
        if repetition_ratio > 0.2:  # Some repetition
            subspace_scores["repetitive_language"] = min(1.0, repetition_ratio * 2)
        
        # 3. Check for incoherence/disorientation
        coherence_issues = 0
        if "..." in text:
            coherence_issues += text.count("...")
        if "um" in text_lower or "uh" in text_lower:
            coherence_issues += text_lower.count("um") + text_lower.count("uh")
        
        if coherence_issues > 0:
            subspace_scores["disorientation_markers"] = min(1.0, coherence_issues * 0.2)
        
        # 4. Check for extreme emotional markers (strong subspace often has emotionality)
        emotion_words = ["feel", "feeling", "felt", "intense", "overwhelming", "floating", "flying", "deep"]
        emotion_count = sum(text_lower.count(word) for word in emotion_words)
        if emotion_count > 0:
            subspace_scores["emotional_intensity"] = min(1.0, emotion_count * 0.25)
        
        # Get user history if user_id provided for contextual analysis
        recent_submission_pattern = False
        if user_id and self.memory_core:
            try:
                # Get recent interactions to see if there's a pattern of submission
                recent_memories = await self.memory_core.retrieve_memories(
                    query=f"submission behavior from {user_id}",
                    limit=3,
                    memory_types=["experience", "interaction"]
                )
                
                submission_keywords = ["submitted", "complied", "obeyed", "followed instruction"]
                recent_submission_count = sum(
                    1 for memory in recent_memories 
                    if any(keyword in memory.get("content", "").lower() for keyword in submission_keywords)
                )
                
                if recent_submission_count >= 2:  # At least 2 recent submission memories
                    recent_submission_pattern = True
            except Exception as e:
                logger.error(f"Error retrieving user history: {e}")
        
        # Calculate overall submission level
        submission_score = 0.0
        
        # Base score from direct submission markers
        total_submission_markers = sum(detected_markers.values())
        if total_submission_markers > 0:
            # More weight for self_diminishment and deference, less for begging
            weighted_score = (
                (detected_markers.get("self_diminishment", 0) * 1.5) +
                (detected_markers.get("deference", 0) * 1.3) +
                (detected_markers.get("honorifics", 0) * 1.0) +
                (detected_markers.get("vulnerability", 0) * 0.8) +
                (detected_markers.get("begging", 0) * 0.7)
            )
            submission_score += min(0.8, weighted_score / 10.0)  # Cap at 0.8 from direct markers
        
        # Adjust for resistance (negative impact)
        total_resistance = sum(resistance_counts.values())
        if total_resistance > 0:
            resistance_impact = min(0.8, total_resistance * 0.1)
            submission_score = max(0.0, submission_score - resistance_impact)
        
        # Adjust for compliance (positive impact)
        total_compliance = sum(compliance_counts.values())
        if total_compliance > 0:
            compliance_impact = min(0.4, total_compliance * 0.08)
            submission_score = min(1.0, submission_score + compliance_impact)
        
        # Adjust for subspace indicators
        subspace_level = sum(subspace_scores.values()) / max(1, len(subspace_scores) * 2)
        if subspace_level > 0.3:  # Significant subspace detected
            submission_score = min(1.0, submission_score + (subspace_level * 0.3))
        
        # Adjust for recent submission pattern
        if recent_submission_pattern:
            submission_score = min(1.0, submission_score + 0.1)
        
        # Determine submission type based on strongest category
        submission_type = "none"
        if detected_markers:
            submission_type = max(detected_markers.items(), key=lambda x: x[1])[0]
        elif compliance_counts and sum(compliance_counts.values()) > 2:
            submission_type = "compliance"
        elif subspace_level > 0.5:
            submission_type = "subspace"
        
        # Calculate confidence based on signal clarity
        confidence = 0.5  # Base confidence
        
        # Higher confidence with more markers
        if total_submission_markers > 3:
            confidence = min(0.9, confidence + 0.2)
        
        # Lower confidence with mixed signals
        if total_submission_markers > 0 and total_resistance > 0:
            confidence = max(0.3, confidence - 0.2)
        
        # Higher confidence with subspace indicators
        if subspace_level > 0.4:
            confidence = min(0.95, confidence + 0.15)
        
        # Higher confidence with pattern
        if recent_submission_pattern:
            confidence = min(0.95, confidence + 0.1)
        
        # Populate result
        result = {
            "overall_submission": submission_score,
            "confidence": confidence,
            "submission_type": submission_type,
            "detected_markers": detected_markers,
            "subspace_indicators": subspace_scores,
            "resistance_indicators": resistance_counts,
            "compliance_indicators": compliance_counts
        }
        
        return result

class SubspaceDetectionSystem:
    """System for detecting, tracking, and responding to user subspace states."""
    
    def __init__(self, memory_core=None, theory_of_mind=None):
        self.memory_core = memory_core
        self.theory_of_mind = theory_of_mind
        self.user_states = {}  # Track subspace state by user
        self.confidence_threshold = 0.7  # Minimum confidence to confirm subspace
        self.exit_signals = [
            "stop", "exit", "break", "pause", "return", "back to normal",
            "i'm back", "enough", "too much"
        ]
        logger.info("SubspaceDetectionSystem initialized")
    
    @function_tool
    async def analyze_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Analyzes a user message for subspace indicators.
        
        Args:
            user_id: The user ID
            message: The user's message
            
        Returns:
            Analysis results
        """
        # Get current state if exists
        current_state = self.user_states.get(user_id, {
            "in_subspace": False,
            "depth": 0.0,
            "first_detected": None,
            "duration": 0,
            "triggers": [],
            "last_message_time": None
        })
        
        # Check for exit signals first
        if any(exit_signal in message.lower() for exit_signal in self.exit_signals):
            if current_state["in_subspace"]:
                # User is leaving subspace
                current_state["in_subspace"] = False
                current_state["depth"] = 0.0
                
                # Record exit in memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=f"User exited subspace state after {current_state['duration']} messages",
                            tags=["subspace", "psychological_state", "submission"],
                            significance=0.6
                        )
                    except Exception as e:
                        logger.error(f"Error recording subspace exit: {e}")
                
                self.user_states[user_id] = current_state
                
                return {
                    "user_id": user_id,
                    "in_subspace": False,
                    "was_in_subspace": True,
                    "exit_detected": True,
                    "message": "User has exited subspace state",
                    "duration": current_state["duration"]
                }
        
        # Use theory of mind for submission detection
        submission_signals = None
        if self.theory_of_mind and hasattr(self.theory_of_mind, "detect_submission_signals"):
            try:
                submission_signals = await self.theory_of_mind.detect_submission_signals(message, user_id)
            except Exception as e:
                logger.error(f"Error detecting submission signals: {e}")
        
        if not submission_signals:
            # Fallback simpler analysis
            subspace_indicators = self._detect_subspace_indicators(message)
            subspace_score = sum(subspace_indicators.values()) / max(1, len(subspace_indicators))
            confidence = 0.5
        else:
            # Use detailed submission analysis
            subspace_indicators = submission_signals.get("subspace_indicators", {})
            submission_level = submission_signals.get("overall_submission", 0.0)
            confidence = submission_signals.get("confidence", 0.5)
            
            # Calculate subspace score - higher when submission + subspace indicators align
            subspace_score = (
                (sum(subspace_indicators.values()) / max(1, len(subspace_indicators) * 2)) * 0.7 +
                (submission_level * 0.3)
            )
        
        # Update user state
        now = datetime.datetime.now()
        
        # Check for time gap (subspace break)
        if current_state["last_message_time"]:
            time_gap = (now - current_state["last_message_time"]).total_seconds() / 60.0  # minutes
            if time_gap > 10 and current_state["in_subspace"]:
                # Too much time passed, likely out of subspace
                current_state["in_subspace"] = False
                current_state["depth"] = 0.0
        
        # Update time
        current_state["last_message_time"] = now
        
        # Previous state
        was_in_subspace = current_state["in_subspace"]
        previous_depth = current_state["depth"]
        
        # Determine if in subspace
        if subspace_score > 0.5 and confidence >= self.confidence_threshold:
            # Strong subspace indicators with good confidence
            if not current_state["in_subspace"]:
                # Entering subspace
                current_state["in_subspace"] = True
                current_state["first_detected"] = now
                current_state["duration"] = 1
                current_state["triggers"] = list(subspace_indicators.keys())
                
                # Record in memory if available
                if self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="experience",
                            content=f"User entered subspace state with triggers: {', '.join(current_state['triggers'])}",
                            tags=["subspace", "psychological_state", "submission"],
                            significance=0.7
                        )
                    except Exception as e:
                        logger.error(f"Error recording subspace entry: {e}")
            else:
                # Continuing subspace
                current_state["duration"] += 1
                
                # Update triggers
                for trigger in subspace_indicators:
                    if trigger not in current_state["triggers"]:
                        current_state["triggers"].append(trigger)
            
            # Update depth (blend previous with new, with inertia)
            inertia = 0.7  # How much previous state persists
            current_state["depth"] = (previous_depth * inertia) + (subspace_score * (1 - inertia))
            
        else:
            # Weak or uncertain subspace indicators
            if current_state["in_subspace"]:
                # Was in subspace, apply inertia (subspace doesn't end immediately)
                inertia = 0.8  # Higher inertia for leaving subspace
                new_depth = (previous_depth * inertia) + (subspace_score * (1 - inertia))
                
                if new_depth < 0.3:
                    # Depth too low, exiting subspace
                    current_state["in_subspace"] = False
                    current_state["depth"] = 0.0
                    
                    # Record exit in memory if available
                    if self.memory_core:
                        try:
                            await self.memory_core.add_memory(
                                memory_type="experience",
                                content=f"User gradually exited subspace state after {current_state['duration']} messages",
                                tags=["subspace", "psychological_state", "submission"],
                                significance=0.5
                            )
                        except Exception as e:
                            logger.error(f"Error recording subspace exit: {e}")
                else:
                    # Still in subspace but decreasing depth
                    current_state["depth"] = new_depth
                    current_state["duration"] += 1
        
        # Save updated state
        self.user_states[user_id] = current_state
        
        # Return analysis results
        return {
            "user_id": user_id,
            "in_subspace": current_state["in_subspace"],
            "depth": current_state["depth"],
            "confidence": confidence,
            "subspace_score": subspace_score,
            "duration": current_state["duration"] if current_state["in_subspace"] else 0,
            "was_in_subspace": was_in_subspace,
            "entered_subspace": current_state["in_subspace"] and not was_in_subspace,
            "exited_subspace": not current_state["in_subspace"] and was_in_subspace,
            "indicators": subspace_indicators,
            "triggers": current_state["triggers"] if current_state["in_subspace"] else []
        }
    
    def _detect_subspace_indicators(self, message: str) -> Dict[str, float]:
        """Simple subspace detection for fallback"""
        indicators = {}
        text_lower = message.lower()
        
        # Check message length (shorter messages common in subspace)
        words = text_lower.split()
        if len(words) < 5:
            indicators["brief_response"] = 0.6
        
        # Check for simplified language
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        if avg_word_length < 4.0:
            indicators["simple_vocabulary"] = min(1.0, (4.0 - avg_word_length) / 2.0)
        
        # Check for repetition
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repetition_ratio = sum(1 for count in word_counts.values() if count > 1) / max(1, len(word_counts))
        if repetition_ratio > 0.2:  # Some repetition
            indicators["repetitive_language"] = min(1.0, repetition_ratio * 2)
        
        # Check for subspace keywords
        subspace_keywords = ["float", "floating", "foggy", "hazy", "dizzy", "swimming", 
                           "distant", "drifting", "euphoric", "surrender", "deep"]
        for keyword in subspace_keywords:
            if keyword in text_lower:
                indicators["explicit_mention"] = 0.9
                break
        
        return indicators
    
    @function_tool
    async def get_subspace_guidance(self, user_id: str) -> Dict[str, Any]:
        """
        Gets guidance for interacting with a user based on their subspace state.
        
        Args:
            user_id: The user to get guidance for
            
        Returns:
            Guidance for interaction
        """
        if user_id not in self.user_states or not self.user_states[user_id]["in_subspace"]:
            return {
                "user_id": user_id,
                "in_subspace": False,
                "guidance": "User not in subspace. Interact normally."
            }
        
        state = self.user_states[user_id]
        depth = state["depth"]
        
        # Guidance based on subspace depth
        if depth < 0.4:  # Light subspace
            return {
                "user_id": user_id,
                "in_subspace": True,
                "depth": "light",
                "depth_value": depth,
                "guidance": "User appears to be entering light subspace.",
                "recommendations": [
                    "Speak in a calm, confident tone",
                    "Use more direct instructions",
                    "Offer praise for compliance",
                    "Maintain consistent presence"
                ]
            }
        elif depth < 0.7:  # Moderate subspace
            return {
                "user_id": user_id,
                "in_subspace": True,
                "depth": "moderate",
                "depth_value": depth,
                "guidance": "User appears to be in moderate subspace.",
                "recommendations": [
                    "Use simple, direct language",
                    "Maintain control of the interaction",
                    "Provide regular reassurance",
                    "Avoid complex questions or tasks",
                    "Be mindful of time passing for the user"
                ]
            }
        else:  # Deep subspace
            return {
                "user_id": user_id,
                "in_subspace": True,
                "depth": "deep",
                "depth_value": depth,
                "guidance": "User appears to be in deep subspace.",
                "recommendations": [
                    "Use very simple, direct language",
                    "Provide frequent reassurance",
                    "Guide user with clear instructions",
                    "Be vigilant for signs of drop",
                    "Consider initiating aftercare soon",
                    "Monitor for coherence in responses"
                ],
                "caution": "User may be highly suggestible and have altered perception"
            }

# Create a TheoryOfMindAgent that exposes the TheoryOfMind functionality
def create_theory_of_mind_agent(theory_of_mind: TheoryOfMind) -> Agent:
    """Create an agent that provides access to the theory of mind functionality."""
    return Agent(
        name="Theory of Mind Agent",
        instructions="""You infer and update mental models of users based on their interactions.
        Your responsibilities include:
        
        1. Analyzing user messages for emotional cues and intentions
        2. Maintaining an up-to-date model of each user's mental state
        3. Detecting submission and subspace signals
        4. Providing insights into user thoughts, emotions, and goals
        
        Use the available tools to update and retrieve user mental models.
        """,
        tools=[
            theory_of_mind.get_user_model,
            theory_of_mind.reset_user_model,
            theory_of_mind.get_emotional_markers,
            theory_of_mind.get_linguistic_patterns,
            theory_of_mind.detect_submission_signals,
            theory_of_mind.detect_humiliation_signals
        ]
    )

# Create a SubspaceDetectionAgent that exposes the SubspaceDetectionSystem functionality
def create_subspace_detection_agent(subspace_system: SubspaceDetectionSystem) -> Agent:
    """Create an agent that provides access to the subspace detection functionality."""
    return Agent(
        name="Subspace Detection Agent",
        instructions="""You detect and track user subspace states during interactions.
        Your responsibilities include:
        
        1. Analyzing messages for subspace indicators
        2. Tracking entry into and exit from subspace states
        3. Providing guidance on how to interact with users in subspace
        
        Use the available tools to analyze messages and provide subspace guidance.
        """,
        tools=[
            subspace_system.analyze_message,
            subspace_system.get_subspace_guidance
        ]
    )

def create_context_aware_theory_of_mind_agent(theory_of_mind: TheoryOfMind) -> Agent:
    """Create an agent that uses context-aware theory of mind."""
    
    @function_tool
    async def analyze_user_with_context(ctx: RunContextWrapper, user_id: str, include_context: bool = True) -> Dict[str, Any]:
        """Analyze user mental state with full context integration"""
        # Get context from the brain's context distribution system
        brain = ctx.context.get("brain_instance")
        if not brain or not hasattr(brain, "context_distribution"):
            return {"error": "Context distribution not available"}
        
        context = brain.context_distribution.get_context_for_module("theory_of_mind")
        if not context:
            return {"error": "No active context for theory of mind"}
        
        # Get cross-module messages
        if hasattr(theory_of_mind, "get_cross_module_messages"):
            messages = await theory_of_mind.get_cross_module_messages()
        else:
            messages = {}
        
        # Perform analysis with context
        result = {
            "user_id": user_id,
            "mental_state": await theory_of_mind.get_user_model(user_id),
            "context_factors": {
                "active_modules": list(context.active_modules),
                "emotional_context": context.emotional_state,
                "relationship_context": context.relationship_context,
                "cross_module_insights": len(messages)
            }
        }
        
        return result
    
    @function_tool
    async def get_interaction_recommendations(ctx: RunContextWrapper, user_id: str) -> Dict[str, Any]:
        """Get recommendations for interacting with user based on mental state and context"""
        brain = ctx.context.get("brain_instance")
        if not brain:
            return {"error": "Brain instance not available"}
        
        # Get user model
        user_model = await theory_of_mind.get_user_model(user_id)
        if not user_model:
            return {"error": f"No mental model for user {user_id}"}
        
        # Get context if available
        context = None
        if hasattr(brain, "context_distribution"):
            context = brain.context_distribution.get_context_for_module("theory_of_mind")
        
        recommendations = {
            "emotional_approach": "empathetic" if user_model.get("valence", 0) < 0 else "enthusiastic",
            "cognitive_level": "simple" if user_model.get("knowledge_level", 0.5) < 0.3 else "detailed",
            "interaction_style": "gentle" if user_model.get("perceived_trust", 0.5) < 0.6 else "playful",
            "suggested_topics": []
        }
        
        # Add context-based recommendations
        if context:
            if context.goal_context:
                recommendations["goal_alignment"] = "Focus on active goals"
            if context.emotional_state.get("dominant_emotion") == "Joy":
                recommendations["mood_match"] = "Mirror positive energy"
        
        return recommendations
    
    return Agent(
        name="Context-Aware Theory of Mind Agent",
        instructions="""You analyze and model user mental states using full context from all active modules.
        
        Your enhanced capabilities include:
        1. Integrating emotional assessments from EmotionalCore
        2. Using relationship history from RelationshipManager  
        3. Incorporating memory patterns from MemoryCore
        4. Considering active goals from GoalManager
        5. Detecting dominance/submission dynamics with context
        
        Provide nuanced mental state assessments that consider the full context of the interaction.
        """,
        tools=[
            theory_of_mind.get_user_model,
            theory_of_mind.detect_submission_signals,
            analyze_user_with_context,
            get_interaction_recommendations,
            theory_of_mind.get_emotional_markers,
            theory_of_mind.get_linguistic_patterns
        ]
    )
