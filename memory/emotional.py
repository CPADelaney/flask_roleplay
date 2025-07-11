# memory/emotional.py

import logging
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import openai
from logic.chatgpt_integration import get_openai_client, build_message_history, safe_json_loads

from .connection import with_transaction, TransactionContext
from .core import Memory, MemoryType, MemorySignificance, UnifiedMemoryManager

logger = logging.getLogger("memory_emotional")

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
    
    # Emotion categories and their properties
    EMOTIONS = {
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
        if primary_emotion not in self.EMOTIONS and primary_emotion.lower() not in self.EMOTIONS:
            primary_emotion = "neutral"
        else:
            primary_emotion = primary_emotion.lower()
            
        # Normalize secondary emotions
        normalized_secondary = {}
        if secondary_emotions:
            for emotion, intensity in secondary_emotions.items():
                if emotion in self.EMOTIONS or emotion.lower() in self.EMOTIONS:
                    normalized_secondary[emotion.lower()] = max(0.0, min(1.0, intensity))
        
        # Calculate emotional_intensity for memory model (0-100 scale)
        valence = self.EMOTIONS[primary_emotion]["valence"] * emotion_intensity
        arousal = self.EMOTIONS[primary_emotion]["arousal"] * emotion_intensity
        
        # Emotional intensity is influenced by arousal and modulated by valence direction
        emotional_intensity = int((abs(valence) * 50) + (arousal * 50))
        
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
            emotional_intensity=emotional_intensity,
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
                    "decay_rate": self.EMOTIONS[primary_emotion]["decay_rate"],
                    "last_update": datetime.now().isoformat()
                }
            },
            timestamp=datetime.now()
        )
        
        memory_id = await memory_manager.add_memory(memory, conn=conn)
        
        # If this is a traumatic memory (high negative emotional intensity), tag it
        if valence < -0.6 and emotion_intensity > 0.7 and significance >= MemorySignificance.HIGH:
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
    
    @with_transaction
    async def analyze_emotional_content(self, 
                                      text: str, 
                                      context: Optional[str] = None,
                                      conn = None) -> Dict[str, Any]:
        """
        Analyze the emotional content of text.
        Can be used before storing a memory to auto-tag emotions.
        
        Args:
            text: The text to analyze
            context: Optional context to help with analysis
            
        Returns:
            Emotional analysis of the text
        """
        try:
            client = get_openai_client()  # Add this line
            context_text = f"\nContext: {context}" if context else ""
            
            prompt = f"""
            Analyze the emotional content of this text:{context_text}
            
            Text: {text}
            
            For the emotional analysis, identify:
            1. The primary emotion present (choose from: joy, sadness, anger, fear, disgust, surprise, anticipation, trust, neutral)
            2. The intensity of that emotion (0.0-1.0)
            3. Any secondary emotions present and their intensities
            4. The overall emotional valence (-1.0 to 1.0, negative to positive)
            5. The arousal level (0.0-1.0, calm to excited)
            
            Format your response as JSON:
            {{
                "primary_emotion": "emotion_name",
                "intensity": 0.X,
                "secondary_emotions": {{"emotion1": 0.X, "emotion2": 0.X}},
                "valence": X.X,
                "arousal": 0.X,
                "explanation": "Brief explanation of your analysis"
            }}
            
            Return only the JSON with no explanation.
            """
            
            response = await client.responses.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You analyze the emotional content of text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            analysis = json.loads(content)
            
            # Validate and normalize the results
            if "primary_emotion" in analysis and analysis["primary_emotion"] not in self.EMOTIONS:
                closest = self._find_closest_emotion(analysis["primary_emotion"])
                analysis["primary_emotion"] = closest
                
            if "secondary_emotions" in analysis:
                normalized = {}
                for emotion, intensity in analysis["secondary_emotions"].items():
                    if emotion in self.EMOTIONS:
                        normalized[emotion] = intensity
                    else:
                        closest = self._find_closest_emotion(emotion)
                        if closest not in normalized:
                            normalized[closest] = intensity
                analysis["secondary_emotions"] = normalized
                
            # Ensure all fields exist
            for field in ["primary_emotion", "intensity", "secondary_emotions", "valence", "arousal"]:
                if field not in analysis:
                    if field == "primary_emotion":
                        analysis[field] = "neutral"
                    elif field == "secondary_emotions":
                        analysis[field] = {}
                    else:
                        analysis[field] = 0.0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing emotional content: {e}")
            # Fallback response
            return {
                "primary_emotion": "neutral",
                "intensity": 0.1,
                "secondary_emotions": {},
                "valence": 0.0,
                "arousal": 0.0,
                "explanation": "Failed to analyze emotional content"
            }
    
    def _find_closest_emotion(self, emotion: str) -> str:
        """Find the closest matching standard emotion."""
        emotion = emotion.lower()
        if emotion in self.EMOTIONS:
            return emotion
            
        # Map common variations
        emotion_map = {
            "happy": "joy",
            "happiness": "joy",
            "excited": "joy",
            "joyful": "joy",
            "joyous": "joy",
            "depressed": "sadness",
            "depressing": "sadness",
            "blue": "sadness",
            "upset": "sadness",
            "sorrowful": "sadness",
            "mad": "anger",
            "furious": "anger",
            "enraged": "anger",
            "hate": "anger",
            "irate": "anger",
            "scared": "fear",
            "terrified": "fear",
            "anxious": "fear",
            "worried": "fear",
            "dread": "fear",
            "repulsed": "disgust",
            "revolted": "disgust",
            "shocked": "surprise",
            "amazed": "surprise",
            "astonished": "surprise",
            "hopeful": "anticipation",
            "eager": "anticipation",
            "confident": "trust",
            "calm": "neutral",
            "indifferent": "neutral"
        }
        
        if emotion in emotion_map:
            return emotion_map[emotion]
            
        # Default to neutral
        return "neutral"
    
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
        primary_emotion = current_mood.get("primary_emotion", "neutral").lower()
        if primary_emotion not in self.EMOTIONS:
            primary_emotion = self._find_closest_emotion(primary_emotion)
            
        # Get mood valence and arousal
        mood_valence = self.EMOTIONS[primary_emotion]["valence"] * current_mood.get("intensity", 0.5)
        mood_arousal = self.EMOTIONS[primary_emotion]["arousal"] * current_mood.get("intensity", 0.5)
        
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
            # Higher when valence matches in sign and magnitude
            valence_match = 1.0 - abs(mood_valence - memory_valence) / 2.0
            
            # Arousal influences recall but less than valence
            arousal_match = 1.0 - abs(mood_arousal - memory_arousal) / 2.0
            
            congruence_score = (valence_match * 0.7) + (arousal_match * 0.3)
            
            # Memories with high emotional intensity are more likely to be recalled
            emotional_boost = memory.emotional_intensity / 100.0
            
            # Recent memories are more likely to be recalled
            recency_factor = 0.0
            if memory.timestamp:
                days_old = (datetime.now() - memory.timestamp).days
                recency_factor = max(0.0, 1.0 - (days_old / 30.0))  # Within 30 days
            
            final_score = (congruence_score * 0.6) + (emotional_boost * 0.2) + (recency_factor * 0.2)
            
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
        
        emotional_state = {}
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
                    "stability": 0.5
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
            old_emotion = emotional_state["current_emotion"].copy() if "current_emotion" in emotional_state else {}
            
            emotional_state["current_emotion"] = {
                "primary": current_emotion.get("primary_emotion", "neutral"),
                "intensity": current_emotion.get("intensity", 0.5),
                "secondary": current_emotion.get("secondary_emotions", {}),
                "valence": current_emotion.get("valence", 0.0),
                "arousal": current_emotion.get("arousal", 0.0),
                "last_update": datetime.now().isoformat(),
                "previous": old_emotion
            }
            
            # Update mood (slower changing)
            if "mood" in emotional_state:
                # Calculate new mood using weighted average
                # Mood changes more slowly than emotions
                mood_inertia = emotional_state["mood"].get("stability", 0.5)
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
            # In a real implementation, this would be more sophisticated
            words = [w for w in memory_text.split() if len(w) > 4]
            if words:
                # Choose a few words as potential triggers
                trigger_count = min(3, len(words))
                triggers = random.sample(words, trigger_count)
                
                for trigger in triggers:
                    if trigger not in [t.get("word") for t in emotional_state["trauma"].get("triggers", [])]:
                        emotional_state["trauma"]["triggers"].append({
                            "word": trigger,
                            "source_event_id": trauma_event.get("memory_id"),
                            "intensity": trauma_event.get("intensity", 0.5),
                            "decay_rate": 0.8  # Triggers can fade over time with positive exposures
                        })
                        
            # Trauma can shift emotional bias
            if "emotional_bias" in emotional_state:
                # Trauma typically increases negative recall bias
                emotional_state["emotional_bias"]["negative_recall_bias"] += 0.1
                emotional_state["emotional_bias"]["emotional_sensitivity"] += 0.05
                
                # Cap values
                emotional_state["emotional_bias"]["negative_recall_bias"] = min(1.0, emotional_state["emotional_bias"]["negative_recall_bias"])
                emotional_state["emotional_bias"]["emotional_sensitivity"] = min(1.0, emotional_state["emotional_bias"]["emotional_sensitivity"])
        
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
            
        emotional_state = row["emotional_state"] if isinstance(row["emotional_state"], dict) else json.loads(row["emotional_state"])
        last_updated = row["last_updated"]
        
        # Calculate time since last update
        time_diff = (datetime.now() - last_updated).total_seconds()
        time_diff_hours = time_diff / 3600.0
        
        # Apply emotional decay if it's been a while
        if time_diff_hours > 1.0 and "current_emotion" in emotional_state:
            decay_factor = min(1.0, time_diff_hours / 12.0)  # Full decay in 12 hours
            current_emotion = emotional_state["current_emotion"]
            
            # Decay intensity towards neutral
            if current_emotion["intensity"] > 0.2:
                new_intensity = max(0.1, current_emotion["intensity"] - (decay_factor * 0.3))
                current_emotion["intensity"] = new_intensity
                
                # Update valence and arousal proportionally
                current_emotion["valence"] *= (new_intensity / current_emotion["intensity"])
                current_emotion["arousal"] *= (new_intensity / current_emotion["intensity"])
                
                # If we've decayed significantly, update in database
                if decay_factor > 0.2:
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
                    # decay_factor ranges from 0.0 (no decay) to 1.0 (full decay)
                    decay_factor = 1.0 - pow(1.0 - decay_rate, time_diff_days)
                    
                    # Calculate new intensity
                    new_intensity = max(0.1, current_intensity - (decay_factor * initial_intensity))
                    
                    # Update memory emotional data
                    emotions_data = metadata.get("emotions", {})
                    primary_data = emotions_data.get("primary", {})
                    
                    if primary_data:
                        primary_data["intensity"] = new_intensity
                        
                    # Update memory's emotional_intensity
                    old_emotional_intensity = memory.emotional_intensity
                    new_emotional_intensity = int(old_emotional_intensity * (new_intensity / current_intensity))
                    
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
            except Exception as e:
                logger.error(f"Error processing emotional decay for memory {memory.id}: {e}")
        
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
