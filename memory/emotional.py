# memory/emotional.py

import logging
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional
from textwrap import dedent
from os import getenv
from difflib import get_close_matches
from pydantic import BaseModel, Field, ValidationError, model_validator

from logic.chatgpt_integration import get_openai_client

from .connection import with_transaction
from .core import Memory, MemoryType, MemorySignificance, UnifiedMemoryManager

logger = logging.getLogger("memory_emotional")

# ======== Constants ========
# Emotional decay and scoring weights
VALENCE_MATCH_WEIGHT = 0.7
AROUSAL_MATCH_WEIGHT = 0.3
CONGRUENCE_EMOTIONAL_BOOST_WEIGHT = 0.2
CONGRUENCE_RECENCY_WEIGHT = 0.2
CONGRUENCE_SCORE_WEIGHT = 0.6

# Trauma thresholds
TRAUMA_VALENCE_THRESHOLD = -0.6
TRAUMA_INTENSITY_THRESHOLD = 0.7

# Decay parameters
FULL_DECAY_HOURS = 12.0
MIN_DECAY_HOURS = 1.0
DECAY_INTENSITY_REDUCTION = 0.3
MIN_EMOTION_INTENSITY = 0.1
DECAY_UPDATE_THRESHOLD = 0.2

# Mood inertia (how slowly mood changes compared to emotions)
DEFAULT_MOOD_INERTIA = 0.5

# ======== Emotion Metadata ========
EMOTION_META = {
    "joy": {"valence": 1.0, "arousal": 0.8, "decay_rate": 0.8, "color": "#FFD700"},
    "sadness": {"valence": -0.8, "arousal": -0.4, "decay_rate": 0.4, "color": "#4682B4"},
    "anger": {"valence": -0.7, "arousal": 0.9, "decay_rate": 0.7, "color": "#FF4500"},
    "fear": {"valence": -0.9, "arousal": 0.8, "decay_rate": 0.5, "color": "#800080"},
    "disgust": {"valence": -0.6, "arousal": 0.5, "decay_rate": 0.6, "color": "#228B22"},
    "surprise": {"valence": 0.4, "arousal": 0.9, "decay_rate": 0.9, "color": "#FF8C00"},
    "anticipation": {"valence": 0.6, "arousal": 0.5, "decay_rate": 0.7, "color": "#FF69B4"},
    "trust": {"valence": 0.9, "arousal": 0.2, "decay_rate": 0.5, "color": "#6495ED"},
    "neutral": {"valence": 0.0, "arousal": 0.0, "decay_rate": 1.0, "color": "#A9A9A9"}
}

# Simple set for validation
EMOTIONS = set(EMOTION_META.keys())

# Emotion synonym mapping
# Most common variations are handled by fuzzy matching in _closest_emotion()
# This explicit mapping is for cases where fuzzy matching might fail or
# where we want to guarantee specific mappings regardless of edit distance
EMOTION_SYNONYMS = {
    "happy": "joy",
    "depressed": "sadness", 
    "mad": "anger",
    "scared": "fear",
    "anxious": "fear",
    "hopeful": "anticipation",
    "confident": "trust",
    "calm": "neutral"
}

# ======== Helper Functions ========
def _clip(x: Any, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range, with type coercion."""
    try:
        x_float = float(x)
        return max(lo, min(hi, x_float))
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to convert '{x}' to float, defaulting to {lo}: {e}", stacklevel=2)
        return lo

def _closest_emotion(term: str) -> str:
    """
    Fuzzy-match an unknown label to the nearest canonical emotion.
    Falls back to 'neutral' when nothing is reasonably close.
    """
    term_lower = term.lower()
    
    # Check direct match first
    if term_lower in EMOTIONS:
        return term_lower
    
    # Check synonyms
    if term_lower in EMOTION_SYNONYMS:
        return EMOTION_SYNONYMS[term_lower]
    
    # Fuzzy match
    matches = get_close_matches(term_lower, EMOTIONS, n=1, cutoff=0.5)
    return matches[0] if matches else "neutral"

# ======== Models ========
class EmotionalAnalysis(BaseModel):
    primary_emotion: str = Field(default="neutral")
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    secondary_emotions: Dict[str, float] = Field(default_factory=dict)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = Field(default="")

    @model_validator(mode="after")
    def normalize(cls, v: "EmotionalAnalysis") -> "EmotionalAnalysis":
        # Normalize primary emotion
        v.primary_emotion = _closest_emotion(v.primary_emotion)

        # Normalize secondary emotions
        clean_secondaries: Dict[str, float] = {}
        for emo, score in v.secondary_emotions.items():
            key = _closest_emotion(emo)
            clean_secondaries[key] = _clip(score, 0.0, 1.0)
        v.secondary_emotions = clean_secondaries

        # Clip all numeric fields
        v.intensity = _clip(v.intensity, 0.0, 1.0)
        v.valence = _clip(v.valence, -1.0, 1.0)
        v.arousal = _clip(v.arousal, 0.0, 1.0)
        return v

# ======== Main Class ========
class EmotionalMemoryManager:
    """
    Advanced emotional memory processing system.
    Handles how emotions influence memory formation, recall, and interpretation.
    
    Features:
    - Emotional tagging of memories
    - Mood-congruent recall
    - Emotional fading and intensification
    - Trauma modeling
    - Positive/negative bias based on emotional state
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    @with_transaction
    async def add_emotional_memory(self,
                                 entity_type: str,
                                 entity_id: int,
                                 memory_text: str,
                                 primary_emotion: str,
                                 emotion_intensity: float,
                                 secondary_emotions: Optional[Dict[str, float]] = None,
                                 significance: int = MemorySignificance.MEDIUM,
                                 tags: List[str] = None,
                                 conn = None) -> Dict[str, Any]:
        """
        Create a memory with rich emotional data.
        
        Args:
            entity_type: Type of entity that owns the memory
            entity_id: ID of the entity
            memory_text: The memory text
            primary_emotion: The main emotion associated with this memory
            emotion_intensity: Intensity of the primary emotion (0.0-1.0)
            secondary_emotions: Dict of secondary emotions and their intensities
            significance: Memory significance
            tags: Additional tags
            
        Returns:
            Created memory information
        """
        # Validate emotion
        primary_emotion = _closest_emotion(primary_emotion)
            
        # Normalize secondary emotions
        normalized_secondary = {}
        if secondary_emotions:
            for emotion, intensity in secondary_emotions.items():
                normalized_emotion = _closest_emotion(emotion)
                normalized_secondary[normalized_emotion] = _clip(intensity, 0.0, 1.0)
        
        # Calculate emotional_intensity for memory model (0-100 scale)
        # Formula: abs(valence) * 50 + arousal * 50
        # This treats positive and negative emotions equally (joy at +1.0 = fear at -1.0)
        # Both contribute to memorability - extreme emotions in either direction are memorable
        # TODO: If business rules change to weight negative emotions differently, adjust formula here
        emotion_meta = EMOTION_META[primary_emotion]
        valence = emotion_meta["valence"] * emotion_intensity
        arousal = emotion_meta["arousal"] * emotion_intensity
        
        # Emotional intensity is influenced by arousal and modulated by valence direction
        # Keep as float for precision
        emotional_intensity = (abs(valence) * 50) + (arousal * 50)
        
        # Create memory manager
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Create memory with rich emotional metadata
        memory = Memory(
            text=memory_text,
            memory_type=MemoryType.OBSERVATION,
            significance=significance,
            emotional_intensity=int(emotional_intensity),  # Convert to int for DB storage
            tags=(tags or []) + ["emotional", primary_emotion],
            metadata={
                "emotions": {
                    "primary": {
                        "name": primary_emotion,
                        "intensity": emotion_intensity
                    },
                    "secondary": normalized_secondary,
                    "valence": valence,
                    "arousal": arousal
                },
                "emotional_decay": {
                    "initial_intensity": emotion_intensity,
                    "current_intensity": emotion_intensity,
                    "decay_rate": emotion_meta["decay_rate"],
                    "last_update": datetime.now().isoformat()
                }
            },
            timestamp=datetime.now()
        )
        
        memory_id = await memory_manager.add_memory(memory, conn=conn)
        
        # If this is a traumatic memory, tag it
        if (valence < TRAUMA_VALENCE_THRESHOLD and 
            emotion_intensity > TRAUMA_INTENSITY_THRESHOLD and 
            significance >= MemorySignificance.HIGH):
            
            # Update metadata to mark as traumatic
            await conn.execute("""
                UPDATE unified_memories
                SET tags = array_append(tags, 'traumatic')
                WHERE id = $1
            """, memory_id)
            
            # Record in the entity's emotional state
            await self.update_entity_emotional_state(
                entity_type=entity_type,
                entity_id=entity_id,
                trauma_event={
                    "memory_id": memory_id,
                    "memory_text": memory_text,
                    "emotion": primary_emotion,
                    "intensity": emotion_intensity,
                    "timestamp": datetime.now().isoformat()
                },
                conn=conn
            )
        
        return {
            "memory_id": memory_id,
            "primary_emotion": primary_emotion,
            "emotion_intensity": emotion_intensity,
            "emotional_intensity": emotional_intensity,
            "valence": valence,
            "arousal": arousal
        }
    
    async def analyze_emotional_content(
        self,
        text: str,
        context: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect the emotional fingerprint of *text* in one Responses-API call.
        Returns a dict matching the EmotionalAnalysis schema.
        """
        model = model or getenv("EMOTION_ANALYSIS_MODEL", "gpt-4o-mini")
    
        try:
            client = get_openai_client()
    
            prompt = dedent(f"""
                Analyze the emotional content of this text:
                {"Context: " + context if context else ""}
    
                Text: {text}
    
                Identify:
                  1. primary_emotion   (joy/sadness/anger/fear/disgust/surprise/anticipation/trust/neutral)
                  2. intensity         (0-1)
                  3. secondary_emotions (dict of emotion:intensity)
                  4. valence           (-1 to 1)
                  5. arousal           (0-1)
    
                Respond **only** with JSON in exactly this shape:
                {{
                  "primary_emotion": "emotion_name",
                  "intensity": 0.0,
                  "secondary_emotions": {{}},
                  "valence": 0.0,
                  "arousal": 0.0,
                  "explanation": "..."
                }}
            """)
    
            # ---- Responses API call (strict JSON output) ----
            resp = client.responses.create(
                model=model,
                instructions=(
                    "You are an emotion-analysis engine. "
                    "Return ONLY the JSON object described—no extra text."
                ),
                input=prompt,
                temperature=0.3,
            )
    
            # Parse and validate with Pydantic
            data = json.loads(resp.output_text)
            parsed = EmotionalAnalysis.model_validate(data)
            return parsed.model_dump(mode="python")
    
        except Exception:
            logger.exception("Emotion analysis failed")
            return EmotionalAnalysis(
                explanation="Fallback: analysis failed"
            ).model_dump(mode="python")

    
    @with_transaction
    async def retrieve_mood_congruent_memories(self,
                                            entity_type: str,
                                            entity_id: int,
                                            current_mood: Dict[str, Any],
                                            limit: int = 5,
                                            conn = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories that match the current emotional state.
        This simulates mood-congruent recall bias in humans.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            current_mood: Dict with at least primary_emotion and intensity
            limit: Maximum number of memories to return
            
        Returns:
            List of mood-congruent memories
        """
        # Create memory manager
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        # Get primary emotion and check if it's valid
        primary_emotion = _closest_emotion(current_mood.get("primary_emotion", "neutral"))
        intensity = current_mood.get("intensity", 0.5)
            
        # Get mood valence and arousal
        emotion_meta = EMOTION_META[primary_emotion]
        mood_valence = emotion_meta["valence"] * intensity
        mood_arousal = emotion_meta["arousal"] * intensity
        
        # Find emotional memories
        memories = await memory_manager.retrieve_memories(
            tags=["emotional"],
            limit=20,  # Get more for filtering
            conn=conn
        )
        
        if not memories:
            return []
            
        # Score memories by mood congruence
        scored_memories = []
        for memory in memories:
            metadata = memory.metadata or {}
            emotions_data = metadata.get("emotions", {})
            
            memory_valence = emotions_data.get("valence", 0.0)
            memory_arousal = emotions_data.get("arousal", 0.0)
            
            # Calculate mood congruence score
            valence_match = 1.0 - abs(mood_valence - memory_valence) / 2.0
            arousal_match = 1.0 - abs(mood_arousal - memory_arousal) / 2.0
            
            congruence_score = (valence_match * VALENCE_MATCH_WEIGHT) + (arousal_match * AROUSAL_MATCH_WEIGHT)
            
            # Memories with high emotional intensity are more likely to be recalled
            emotional_boost = memory.emotional_intensity / 100.0
            
            # Recent memories are more likely to be recalled
            recency_factor = 0.0
            if memory.timestamp:
                days_old = (datetime.now() - memory.timestamp).days
                recency_factor = max(0.0, 1.0 - (days_old / 30.0))  # Within 30 days
            
            final_score = (congruence_score * CONGRUENCE_SCORE_WEIGHT) + \
                         (emotional_boost * CONGRUENCE_EMOTIONAL_BOOST_WEIGHT) + \
                         (recency_factor * CONGRUENCE_RECENCY_WEIGHT)
            
            scored_memories.append((memory, final_score))
        
        # Sort by score and return top matches
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [
            {
                "id": memory.id,
                "text": memory.text,
                "primary_emotion": memory.metadata.get("emotions", {}).get("primary", {}).get("name", "neutral"),
                "emotional_intensity": memory.emotional_intensity,
                "congruence_score": score,
                "timestamp": memory.timestamp.isoformat() if memory.timestamp else None
            }
            for memory, score in scored_memories[:limit]
        ]
        
        return top_memories
    
    @with_transaction
    async def update_entity_emotional_state(self,
                                         entity_type: str,
                                         entity_id: int,
                                         current_emotion: Optional[Dict[str, Any]] = None,
                                         trauma_event: Optional[Dict[str, Any]] = None,
                                         conn = None) -> Dict[str, Any]:
        """
        Update the emotional state of an entity.
        This state affects memory recall, formation, and interpretation.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            current_emotion: Current emotional state (if updating)
            trauma_event: Trauma event (if recording trauma)
            
        Returns:
            Updated emotional state
        """
        # Get current state first
        row = await conn.fetchrow("""
            SELECT emotional_state
            FROM EntityEmotionalState
            WHERE user_id = $1 AND conversation_id = $2 AND entity_type = $3 AND entity_id = $4
        """, self.user_id, self.conversation_id, entity_type, entity_id)
        
        emotional_state: Dict[str, Any] = {}
        if row and row["emotional_state"]:
            emotional_state = row["emotional_state"] if isinstance(row["emotional_state"], dict) else json.loads(row["emotional_state"])
        else:
            # Initialize with defaults
            emotional_state = {
                "current_emotion": {
                    "primary": "neutral",
                    "intensity": 0.1,
                    "secondary": {},
                    "valence": 0.0,
                    "arousal": 0.0,
                    "last_update": datetime.now().isoformat()
                },
                "mood": {
                    "baseline_valence": 0.0,
                    "baseline_arousal": 0.1,
                    "current_valence": 0.0,
                    "current_arousal": 0.1,
                    "stability": DEFAULT_MOOD_INERTIA
                },
                "trauma": {
                    "events": [],
                    "triggers": []
                },
                "emotional_bias": {
                    "positive_recall_bias": 0.0,
                    "negative_recall_bias": 0.0,
                    "emotional_sensitivity": 0.5
                }
            }
        
        # Update current emotion if provided
        if current_emotion:
            old_emotion = emotional_state.get("current_emotion", {}).copy()
            
            # Normalize the emotion
            primary = _closest_emotion(current_emotion.get("primary_emotion", "neutral"))
            
            emotional_state["current_emotion"] = {
                "primary": primary,
                "intensity": current_emotion.get("intensity", 0.5),
                "secondary": current_emotion.get("secondary_emotions", {}),
                "valence": current_emotion.get("valence", 0.0),
                "arousal": current_emotion.get("arousal", 0.0),
                "last_update": datetime.now().isoformat(),
                "previous": old_emotion
            }
            
            # Update mood (slower changing)
            if "mood" in emotional_state:
                mood_inertia = emotional_state["mood"].get("stability", DEFAULT_MOOD_INERTIA)
                new_valence = (emotional_state["mood"]["current_valence"] * mood_inertia) + \
                              (current_emotion.get("valence", 0.0) * (1.0 - mood_inertia))
                new_arousal = (emotional_state["mood"]["current_arousal"] * mood_inertia) + \
                              (current_emotion.get("arousal", 0.0) * (1.0 - mood_inertia))
                
                emotional_state["mood"]["current_valence"] = new_valence
                emotional_state["mood"]["current_arousal"] = new_arousal
        
        # Add trauma event if provided
        if trauma_event:
            if "trauma" not in emotional_state:
                emotional_state["trauma"] = {"events": [], "triggers": []}
                
            emotional_state["trauma"]["events"].append(trauma_event)
            
            # Extract potential triggers from the trauma event
            memory_text = trauma_event.get("memory_text", "")
            
            # Generate potential triggers (keywords from the memory)
            words = [w for w in memory_text.split() if len(w) > 4]
            if words:
                trigger_count = min(3, len(words))
                # NOTE: For deterministic tests, seed random before calling this method
                # e.g., random.seed(42) in test setup, or inject a custom RNG
                triggers = random.sample(words, trigger_count)
                
                existing_trigger_words = {t.get("word") for t in emotional_state["trauma"].get("triggers", [])}
                
                for trigger in triggers:
                    if trigger not in existing_trigger_words:
                        emotional_state["trauma"]["triggers"].append({
                            "word": trigger,
                            "source_event_id": trauma_event.get("memory_id"),
                            "intensity": trauma_event.get("intensity", 0.5),
                            "decay_rate": 0.8
                        })
                        
            # Trauma can shift emotional bias
            if "emotional_bias" in emotional_state:
                # Trauma typically increases negative recall bias
                emotional_state["emotional_bias"]["negative_recall_bias"] = min(
                    1.0, emotional_state["emotional_bias"]["negative_recall_bias"] + 0.1
                )
                emotional_state["emotional_bias"]["emotional_sensitivity"] = min(
                    1.0, emotional_state["emotional_bias"]["emotional_sensitivity"] + 0.05
                )
        
        # Save updated state
        await conn.execute("""
            INSERT INTO EntityEmotionalState (
                user_id, conversation_id, entity_type, entity_id, emotional_state, last_updated
            )
            VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, conversation_id, entity_type, entity_id)
            DO UPDATE SET 
                emotional_state = $5,
                last_updated = CURRENT_TIMESTAMP
        """, self.user_id, self.conversation_id, entity_type, entity_id, json.dumps(emotional_state))
        
        return emotional_state
    
    @with_transaction
    async def get_entity_emotional_state(self,
                                      entity_type: str,
                                      entity_id: int,
                                      conn = None) -> Dict[str, Any]:
        """
        Get the current emotional state of an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Current emotional state
        """
        row = await conn.fetchrow("""
            SELECT emotional_state, last_updated
            FROM EntityEmotionalState
            WHERE user_id = $1 AND conversation_id = $2 AND entity_type = $3 AND entity_id = $4
        """, self.user_id, self.conversation_id, entity_type, entity_id)
        
        if not row:
            return {
                "error": "No emotional state found",
                "default_state": True,
                "current_emotion": {
                    "primary": "neutral",
                    "intensity": 0.1
                }
            }
            
        emotional_state: Dict[str, Any] = row["emotional_state"] if isinstance(row["emotional_state"], dict) else json.loads(row["emotional_state"])
        last_updated = row["last_updated"]
        
        # Calculate time since last update
        time_diff = (datetime.now() - last_updated).total_seconds()
        time_diff_hours = time_diff / 3600.0
        
        # Apply emotional decay if it's been a while
        if time_diff_hours > MIN_DECAY_HOURS and "current_emotion" in emotional_state:
            decay_factor = min(1.0, time_diff_hours / FULL_DECAY_HOURS)
            current_emotion = emotional_state["current_emotion"]
            
            # Store old intensity before mutating
            old_intensity = current_emotion.get("intensity", 0.1)
            
            # Decay intensity towards neutral
            if old_intensity > MIN_EMOTION_INTENSITY * 2:
                new_intensity = max(MIN_EMOTION_INTENSITY, old_intensity - (decay_factor * DECAY_INTENSITY_REDUCTION))
                current_emotion["intensity"] = new_intensity
                
                # Update valence and arousal proportionally
                if old_intensity > 0:  # Avoid div-by-zero
                    scale_factor = new_intensity / old_intensity
                    current_emotion["valence"] = current_emotion.get("valence", 0.0) * scale_factor
                    current_emotion["arousal"] = current_emotion.get("arousal", 0.0) * scale_factor
                
                # If we've decayed significantly, update in database
                if decay_factor > DECAY_UPDATE_THRESHOLD:
                    emotional_state["current_emotion"] = current_emotion
                    await conn.execute("""
                        UPDATE EntityEmotionalState
                        SET emotional_state = $1
                        WHERE user_id = $2 AND conversation_id = $3 AND entity_type = $4 AND entity_id = $5
                    """, json.dumps(emotional_state), self.user_id, self.conversation_id, entity_type, entity_id)
        
        return emotional_state
    
    @with_transaction
    async def process_traumatic_triggers(self,
                                      entity_type: str,
                                      entity_id: int,
                                      text: str,
                                      conn = None) -> Dict[str, Any]:
        """
        Check if text contains triggers for traumatic memories.
        If so, those memories might be more likely to be recalled.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            text: Text to check for triggers
            
        Returns:
            Triggered traumatic memories, if any
        """
        # Get entity's emotional state
        emotional_state = await self.get_entity_emotional_state(
            entity_type=entity_type,
            entity_id=entity_id,
            conn=conn
        )
        
        if "trauma" not in emotional_state or not emotional_state["trauma"].get("triggers"):
            return {"triggered": False}
            
        # Check for triggers
        triggered_events = []
        text_lower = text.lower()
        
        for trigger in emotional_state["trauma"]["triggers"]:
            trigger_word = trigger.get("word", "").lower()
            if trigger_word and trigger_word in text_lower:
                source_event_id = trigger.get("source_event_id")
                
                if source_event_id:
                    # Find the source trauma event
                    for event in emotional_state["trauma"].get("events", []):
                        if event.get("memory_id") == source_event_id:
                            triggered_events.append({
                                "trigger_word": trigger_word,
                                "memory_id": source_event_id,
                                "memory_text": event.get("memory_text", ""),
                                "emotion": event.get("emotion", "fear"),
                                "intensity": trigger.get("intensity", 0.5)
                            })
                            break
        
        if not triggered_events:
            return {"triggered": False}
            
        # Get the trauma memories
        memory_ids = [event["memory_id"] for event in triggered_events]
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        triggered_memories = []
        for memory_id in memory_ids:
            memories = await memory_manager.retrieve_memories(
                exclude_ids=[m["id"] for m in triggered_memories],
                min_significance=MemorySignificance.MEDIUM,
                limit=1,
                conn=conn
            )
            
            if memories:
                triggered_memories.extend(memories)
        
        # Update entity's emotional state to reflect the triggered trauma
        current_emotion = {
            "primary_emotion": "fear",  # Default trauma response
            "intensity": max([event["intensity"] for event in triggered_events]),
            "valence": -0.8,  # Negative valence for fear
            "arousal": 0.9,   # High arousal for fear
            "secondary_emotions": {
                "anxiety": 0.8,
                "sadness": 0.5
            }
        }
        
        await self.update_entity_emotional_state(
            entity_type=entity_type,
            entity_id=entity_id,
            current_emotion=current_emotion,
            conn=conn
        )
        
        return {
            "triggered": True,
            "trigger_count": len(triggered_events),
            "trigger_words": [event["trigger_word"] for event in triggered_events],
            "triggered_memories": [
                {
                    "id": memory.id,
                    "text": memory.text,
                    "emotional_intensity": memory.emotional_intensity,
                    "significance": memory.significance
                }
                for memory in triggered_memories
            ],
            "emotional_response": current_emotion
        }
    
    @with_transaction
    async def emotional_decay_maintenance(self,
                                       entity_type: str,
                                       entity_id: int,
                                       conn = None) -> Dict[str, Any]:
        """
        Apply emotional decay to memories over time.
        Some emotions fade faster than others.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Results of the decay process
        """
        # Get all emotional memories
        memory_manager = UnifiedMemoryManager(
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        emotional_memories = await memory_manager.retrieve_memories(
            tags=["emotional"],
            limit=100,  # Process a batch at a time
            conn=conn
        )
        
        if not emotional_memories:
            return {"processed": 0}
            
        processed_count = 0
        for memory in emotional_memories:
            metadata = memory.metadata or {}
            decay_data = metadata.get("emotional_decay", {})
            
            if not decay_data:
                continue
                
            # Check if this memory should be decayed
            initial_intensity = decay_data.get("initial_intensity", 0.5)
            current_intensity = decay_data.get("current_intensity", initial_intensity)
            decay_rate = decay_data.get("decay_rate", 0.5)
            last_update_str = decay_data.get("last_update")
            
            if not last_update_str:
                continue
                
            try:
                last_update = datetime.fromisoformat(last_update_str)
                time_diff = (datetime.now() - last_update).total_seconds()
                time_diff_days = time_diff / 86400.0
                
                # Apply decay if it's been at least a day
                if time_diff_days >= 1.0:
                    # Emotions decay exponentially
                    decay_factor = 1.0 - pow(1.0 - decay_rate, time_diff_days)
                    
                    # Calculate new intensity
                    new_intensity = max(MIN_EMOTION_INTENSITY, current_intensity - (decay_factor * initial_intensity))
                    
                    # Update memory emotional data
                    emotions_data = metadata.get("emotions", {})
                    primary_data = emotions_data.get("primary", {})
                    
                    if primary_data:
                        primary_data["intensity"] = new_intensity
                        
                    # Update memory's emotional_intensity (keep as float internally for precision)
                    old_emotional_intensity = memory.emotional_intensity
                    if current_intensity > 0:  # Avoid div-by-zero
                        new_emotional_intensity_float = old_emotional_intensity * (new_intensity / current_intensity)
                        new_emotional_intensity = int(round(new_emotional_intensity_float))
                    else:
                        new_emotional_intensity = int(MIN_EMOTION_INTENSITY * 100)
                    
                    # Update decay data
                    decay_data["current_intensity"] = new_intensity
                    decay_data["last_update"] = datetime.now().isoformat()
                    
                    # Update memory
                    metadata["emotions"] = emotions_data
                    metadata["emotional_decay"] = decay_data
                    
                    await conn.execute("""
                        UPDATE unified_memories
                        SET metadata = $1,
                            emotional_intensity = $2
                        WHERE id = $3
                    """, json.dumps(metadata), new_emotional_intensity, memory.id)
                    
                    processed_count += 1
            except Exception:
                logger.exception(f"Error processing emotional decay for memory {memory.id}")
        
        return {
            "processed": processed_count,
            "total_emotional_memories": len(emotional_memories)
        }

# Create the necessary tables if they don't exist
async def create_emotional_tables():
    """Create the necessary tables for the emotional memory system if they don't exist."""
    from db.connection import get_db_connection_context
    
    async with get_db_connection_context() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS EntityEmotionalState (
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                emotional_state JSONB NOT NULL,
                last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, conversation_id, entity_type, entity_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_entity_emotional_state_lookup 
            ON EntityEmotionalState(user_id, conversation_id, entity_type, entity_id);
            
            CREATE INDEX IF NOT EXISTS idx_entity_emotional_state_updated 
            ON EntityEmotionalState(last_updated);
        """)
