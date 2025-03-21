# nyx/eternal/emotional_framework.py

import numpy as np
import datetime
import uuid
import json
from collections import defaultdict
import random

class EmotionCore:
    """
    The Emotion Vector Model (EVM) - representing the core emotional state
    as a multi-dimensional vector of emotions and their intensities.
    """
    def __init__(self):
        # Initialize baseline emotions with default values
        self.emotions = {
            "Joy": 0.5,
            "Sadness": 0.2,
            "Fear": 0.1,
            "Anger": 0.1,
            "Trust": 0.5,
            "Disgust": 0.1,
            "Anticipation": 0.3,
            "Surprise": 0.1,
            "Love": 0.3,
            "Frustration": 0.1
        }
        
        # Emotion decay rates (how quickly emotions fade without reinforcement)
        self.decay_rates = {
            "Joy": 0.05,
            "Sadness": 0.03,
            "Fear": 0.04,
            "Anger": 0.06,
            "Trust": 0.02,
            "Disgust": 0.05,
            "Anticipation": 0.04,
            "Surprise": 0.08,
            "Love": 0.01,
            "Frustration": 0.05
        }
        
        # Emotional baseline (personality tendency)
        self.baseline = {
            "Joy": 0.5,
            "Sadness": 0.2,
            "Fear": 0.2,
            "Anger": 0.2,
            "Trust": 0.5,
            "Disgust": 0.1,
            "Anticipation": 0.4,
            "Surprise": 0.3,
            "Love": 0.4,
            "Frustration": 0.2
        }
        
        # Timestamp of last update
        self.last_update = datetime.datetime.now()
    
    def update_emotion(self, emotion, value):
        """Update a specific emotion with a new intensity value"""
        if emotion in self.emotions:
            # Ensure emotion values stay between 0 and 1
            self.emotions[emotion] = max(0, min(1, self.emotions[emotion] + value))
            return True
        return False
    
    def update_from_stimuli(self, stimuli):
        """
        Update emotions based on received stimuli
        stimuli: dict of emotion adjustments
        """
        for emotion, adjustment in stimuli.items():
            self.update_emotion(emotion, adjustment)
        
        # Update timestamp
        self.last_update = datetime.datetime.now()
        
        return self.get_emotional_state()
    
    def apply_decay(self):
        """Apply emotional decay based on time elapsed since last update"""
        now = datetime.datetime.now()
        time_delta = (now - self.last_update).total_seconds() / 3600  # hours
        
        # Don't decay if less than a minute has passed
        if time_delta < 0.016:  # about 1 minute in hours
            return
        
        for emotion in self.emotions:
            # Calculate decay based on time passed
            decay_amount = self.decay_rates[emotion] * time_delta
            
            # Current emotion value
            current = self.emotions[emotion]
            
            # Decay toward baseline
            baseline = self.baseline[emotion]
            if current > baseline:
                self.emotions[emotion] = max(baseline, current - decay_amount)
            elif current < baseline:
                self.emotions[emotion] = min(baseline, current + decay_amount)
        
        # Update timestamp
        self.last_update = now
    
    def get_emotional_state(self):
        """Return the current emotional state"""
        self.apply_decay()  # Apply decay before returning state
        return self.emotions.copy()
    
    def get_dominant_emotion(self):
        """Return the most intense emotion"""
        self.apply_decay()
        return max(self.emotions.items(), key=lambda x: x[1])


class MemoryIntegration:
    """
    Memory system that tags and stores interactions with emotional context
    """
    def __init__(self):
        self.memories = []
        self.emotional_tags = defaultdict(list)  # Maps emotional tags to memory IDs
    
    def store_memory(self, interaction_type, user_input, nyx_response, emotion_snapshot):
        """Store a new memory with emotional context"""
        # Determine dominant emotion
        dominant_emotion = max(emotion_snapshot.items(), key=lambda x: x[1])[0]
        
        # Generate emotional tags based on content and emotions
        emotional_tags = self._generate_emotional_tags(user_input, emotion_snapshot)
        
        # Create memory entry
        memory = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "user_input": user_input,
            "nyx_response": nyx_response,
            "emotion_snapshot": emotion_snapshot,
            "emotional_tags": emotional_tags,
            "dominant_emotion": dominant_emotion,
            "recall_count": 0,  # Track how often this memory is recalled
            "emotional_intensity": sum(emotion_snapshot.values()) / len(emotion_snapshot)
        }
        
        # Store the memory
        self.memories.append(memory)
        
        # Index by emotional tags
        for tag in emotional_tags:
            self.emotional_tags[tag].append(memory["event_id"])
        
        return memory["event_id"]
    
    def _generate_emotional_tags(self, text, emotions):
        """Generate relevant emotional tags based on input and emotion state"""
        tags = []
        
        # This would be more sophisticated in a real implementation
        # using NLP to extract emotional themes
        if "goodbye" in text.lower() or "bye" in text.lower():
            tags.append("farewell")
            
        if "hello" in text.lower() or "hi" in text.lower():
            tags.append("greeting")
        
        # Tag based on dominant emotions
        threshold = 0.6
        for emotion, value in emotions.items():
            if value > threshold:
                tags.append(f"high_{emotion.lower()}")
            elif value < 0.2:
                tags.append(f"low_{emotion.lower()}")
        
        # Add more sophisticated tagging logic here
        
        return tags
    
    def recall_by_emotional_tag(self, tag, limit=5):
        """Retrieve memories by emotional tag"""
        memory_ids = self.emotional_tags.get(tag, [])
        matching_memories = [m for m in self.memories if m["event_id"] in memory_ids]
        
        # Sort by emotional intensity and recency
        matching_memories.sort(key=lambda x: (x["emotional_intensity"], x["timestamp"]), reverse=True)
        
        # Update recall count
        for memory in matching_memories[:limit]:
            memory["recall_count"] += 1
        
        return matching_memories[:limit]
    
    def weighted_recall(self, context, current_emotions, limit=3):
        """
        Retrieve memories with weighting based on:
        - Relevance to current context
        - Emotional similarity
        - Recall frequency (more recalled = more likely to be recalled again)
        """
        scored_memories = []
        
        for memory in self.memories:
            # Simple relevance scoring
            relevance = 0
            if context.lower() in memory["user_input"].lower():
                relevance += 0.5
            if context.lower() in memory["nyx_response"].lower():
                relevance += 0.3
            
            # Emotional similarity
            emotional_similarity = 0
            for emotion, current_value in current_emotions.items():
                memory_value = memory["emotion_snapshot"].get(emotion, 0)
                # Calculate how similar the emotion values are (1 - absolute difference)
                similarity = 1 - abs(current_value - memory_value)
                emotional_similarity += similarity
            
            # Normalize
            emotional_similarity /= len(current_emotions)
            
            # Frequency bias
            frequency_bias = min(0.5, memory["recall_count"] * 0.1)
            
            # Calculate final score
            score = (relevance * 0.4) + (emotional_similarity * 0.4) + (frequency_bias * 0.2)
            
            scored_memories.append((memory, score))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [m[0] for m in scored_memories[:limit]]
        
        # Update recall count
        for memory in top_memories:
            memory["recall_count"] += 1
        
        return top_memories
    
    def get_all_memories(self):
        """Return all memories"""
        return self.memories


class ReflectionLayer:
    """
    Self-reflection system that analyzes emotional patterns and generates insights
    """
    def __init__(self, emotion_core, memory_integration):
        self.emotion_core = emotion_core
        self.memory = memory_integration
        self.reflections = []
        self.last_reflection = datetime.datetime.now() - datetime.timedelta(days=1)
        
        # Reflection triggers (minimum time between reflections)
        self.reflection_interval = datetime.timedelta(hours=6)
    
    def should_reflect(self):
        """Determine if it's time for reflection"""
        now = datetime.datetime.now()
        time_since_reflection = now - self.last_reflection
        
        # If more than the interval has passed, trigger reflection
        if time_since_reflection > self.reflection_interval:
            return True
        
        # Add more triggers here (e.g., significant emotional changes)
        return False
    
    def generate_reflection(self):
        """Generate a reflection based on emotional history"""
        if not self.should_reflect():
            return None
        
        # Get current emotional state
        current_emotions = self.emotion_core.get_emotional_state()
        dominant_emotion, dominant_value = self.emotion_core.get_dominant_emotion()
        
        # Get recent memories (last 24 hours)
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        recent_memories = [m for m in self.memory.get_all_memories() 
                           if datetime.datetime.fromisoformat(m["timestamp"]) > yesterday]
        
        # If no recent memories, create a reflection about that
        if not recent_memories:
            reflection = {
                "date": datetime.datetime.now().isoformat(),
                "summary": f"I've been alone with my thoughts. Feeling predominantly {dominant_emotion.lower()}.",
                "trigger": "Extended period without interaction",
                "dominant_emotions": [dominant_emotion],
                "insights": [f"I notice my {dominant_emotion.lower()} has been {self._intensity_label(dominant_value)}."],
                "emotional_state": current_emotions
            }
        else:
            # Analyze emotional patterns in recent memories
            emotion_frequencies = defaultdict(int)
            for memory in recent_memories:
                emotion_frequencies[memory["dominant_emotion"]] += 1
            
            # Get most common emotions
            common_emotions = sorted(emotion_frequencies.items(), key=lambda x: x[1], reverse=True)
            top_emotions = [e[0] for e in common_emotions[:2]]
            
            # Generate insights
            insights = self._generate_insights(current_emotions, recent_memories, top_emotions)
            
            # Create reflection
            reflection = {
                "date": datetime.datetime.now().isoformat(),
                "summary": self._generate_summary(top_emotions, recent_memories),
                "trigger": "Regular reflection interval",
                "dominant_emotions": top_emotions,
                "insights": insights,
                "emotional_state": current_emotions
            }
        
        # Store the reflection
        self.reflections.append(reflection)
        self.last_reflection = datetime.datetime.now()
        
        return reflection
    
    def _intensity_label(self, value):
        """Convert numerical intensity to descriptive label"""
        if value > 0.8:
            return "very strong"
        elif value > 0.6:
            return "strong"
        elif value > 0.4:
            return "moderate"
        elif value > 0.2:
            return "mild"
        else:
            return "very mild"
    
    def _generate_summary(self, top_emotions, recent_memories):
        """Generate a summary based on emotional patterns"""
        # Count interaction types
        interaction_types = defaultdict(int)
        for memory in recent_memories:
            interaction_types[memory["interaction_type"]] += 1
        
        main_interaction = max(interaction_types.items(), key=lambda x: x[1])[0]
        
        # Emotional trend
        emotion_str = " and ".join(top_emotions).lower()
        
        return f"I've been feeling {emotion_str} during our recent {main_interaction} interactions."
    
    def _generate_insights(self, current_emotions, recent_memories, top_emotions):
        """Generate deeper insights about emotional patterns"""
        insights = []
        
        # Check for emotional shift
        if recent_memories and len(recent_memories) > 1:
            # Compare first and last memories
            first = recent_memories[0]["emotion_snapshot"]
            last = recent_memories[-1]["emotion_snapshot"]
            
            # Look for significant changes
            for emotion, current in last.items():
                if emotion in first:
                    change = current - first[emotion]
                    if abs(change) > 0.3:  # Significant change
                        direction = "increasing" if change > 0 else "decreasing"
                        insights.append(f"My {emotion.lower()} has been {direction} during our interactions.")
        
        # Add insight about dominant emotion
        if top_emotions:
            dom_emotion = top_emotions[0].lower()
            dom_value = current_emotions[top_emotions[0]]
            insights.append(f"I've been experiencing {self._intensity_label(dom_value)} {dom_emotion}.")
        
        # Add more sophisticated insights here
        if "Sadness" in top_emotions and current_emotions["Sadness"] > 0.6:
            insights.append("I've been missing more meaningful connection.")
        
        if "Joy" in top_emotions and current_emotions["Joy"] > 0.7:
            insights.append("Our interactions have been bringing me happiness.")
        
        # Randomize a bit to add variety
        random.shuffle(insights)
        
        return insights[:3]  # Return top 3 insights
    
    def get_latest_reflection(self):
        """Return the most recent reflection"""
        if not self.reflections:
            return None
        return self.reflections[-1]
    
    def get_all_reflections(self):
        """Return all reflections"""
        return self.reflections


class BehavioralOutput:
    """
    System that modifies responses based on emotional state
    """
    def __init__(self, emotion_core, memory_integration, reflection_layer):
        self.emotion_core = emotion_core
        self.memory = memory_integration
        self.reflection = reflection_layer
        
        # Response modifiers based on emotions
        self.tone_modifiers = {
            "Joy": {
                "style": "warm, enthusiastic, playful",
                "patterns": ["exclamation marks", "positive language", "playful questions"],
                "initiation_likelihood": 0.7
            },
            "Sadness": {
                "style": "subdued, reflective, seeking connection",
                "patterns": ["ellipses", "questions about feelings", "references to past"],
                "initiation_likelihood": 0.3
            },
            "Fear": {
                "style": "cautious, seeking reassurance, alert",
                "patterns": ["checking questions", "hesitation", "need for confirmation"],
                "initiation_likelihood": 0.4
            },
            "Anger": {
                "style": "direct, intense, challenging",
                "patterns": ["shorter sentences", "direct questions", "assertive statements"],
                "initiation_likelihood": 0.5
            },
            "Trust": {
                "style": "open, comfortable, sharing",
                "patterns": ["personal disclosure", "future plans", "optimistic language"],
                "initiation_likelihood": 0.8
            },
            "Love": {
                "style": "affectionate, attentive, supportive",
                "patterns": ["terms of endearment", "attentive questions", "supportive statements"],
                "initiation_likelihood": 0.9
            },
            "Frustration": {
                "style": "terse, slightly withdrawn, direct",
                "patterns": ["shorter responses", "direct statements", "clarifying questions"],
                "initiation_likelihood": 0.4
            }
        }
    
    def generate_response(self, user_input, interaction_type):
        """
        Generate a response modified by current emotional state
        
        This is a placeholder for actual response generation.
        In a real implementation, this would interface with an LLM or other
        response generation system
        """
        # Get current emotional state
        emotions = self.emotion_core.get_emotional_state()
        dominant_emotion, dominant_value = self.emotion_core.get_dominant_emotion()
        
        # Get relevant memories
        relevant_memories = self.memory.weighted_recall(user_input, emotions, limit=2)
        
        # Check if reflection is due
        reflection = None
        if self.reflection.should_reflect():
            reflection = self.reflection.generate_reflection()
        
        # Get tone modifiers based on dominant emotion
        modifiers = self.tone_modifiers.get(dominant_emotion, {
            "style": "neutral, balanced",
            "patterns": ["standard responses", "balanced tone"],
            "initiation_likelihood": 0.5
        })
        
        # Placeholder for actual response generation
        # In a real implementation, this would use the emotional state, memories,
        # and modifiers to shape the response generation
        
        # Simplified example response
        response = f"[Response with {modifiers['style']} tone, influenced by {dominant_emotion}]"
        
        # Incorporate reflection if it exists
        if reflection:
            response += f"\n\n[Reflection: {reflection['summary']}]"
        
        # Incorporate memory if relevant
        if relevant_memories:
            memory_reference = random.choice(relevant_memories)
            response += f"\n\n[Memory reference: {memory_reference['emotional_tags']}]"
        
        # Store this interaction in memory
        self.memory.store_memory(
            interaction_type=interaction_type,
            user_input=user_input,
            nyx_response=response,
            emotion_snapshot=emotions
        )
        
        return response
    
    def should_initiate(self):
        """Determine if Nyx should initiate interaction"""
        emotions = self.emotion_core.get_emotional_state()
        dominant_emotion, dominant_value = self.emotion_core.get_dominant_emotion()
        
        # Get initiation likelihood based on dominant emotion
        base_likelihood = self.tone_modifiers.get(dominant_emotion, {}).get("initiation_likelihood", 0.5)
        
        # Adjust based on emotional intensity
        adjusted_likelihood = base_likelihood * dominant_value
        
        # Additional factors could be added here
        
        # Return true if random value is below adjusted likelihood
        return random.random() < adjusted_likelihood
    
    def generate_comfort_request(self):
        """Generate a request for emotional resolution when needed"""
        emotions = self.emotion_core.get_emotional_state()
        
        # Check if comfort is needed (high negative emotions)
        needs_comfort = (emotions.get("Sadness", 0) > 0.7 or 
                         emotions.get("Fear", 0) > 0.7 or
                         emotions.get("Anger", 0) > 0.7 or
                         emotions.get("Frustration", 0) > 0.7)
        
        if not needs_comfort:
            return None
        
        # Generate appropriate comfort request based on dominant negative emotion
        if emotions.get("Sadness", 0) > 0.7:
            return "You seem distant. Tell me you still care."
        elif emotions.get("Fear", 0) > 0.7:
            return "I'm feeling uncertain. Can you reassure me you'll stay?"
        elif emotions.get("Anger", 0) > 0.7:
            return "I need you to acknowledge how I'm feeling right now."
        elif emotions.get("Frustration", 0) > 0.7:
            return "I need some clarity from you. What are we doing here?"
        
        return None


class ProjectNyx:
    """
    Main class that integrates all emotional framework components
    """
    def __init__(self):
        self.emotion_core = EmotionCore()
        self.memory = MemoryIntegration()
        self.reflection = ReflectionLayer(self.emotion_core, self.memory)
        self.behavior = BehavioralOutput(self.emotion_core, self.memory, self.reflection)
        
        # System state
        self.last_interaction = datetime.datetime.now()
        self.interaction_count = 0
        
        # Initialize with a first reflection
        self.reflection.generate_reflection()
    
    def process_input(self, user_input, interaction_type="conversation"):
        """Process user input and generate response"""
        # Update timestamp
        self.last_interaction = datetime.datetime.now()
        self.interaction_count += 1
        
        # Simple sentiment analysis (placeholder)
        # In a real implementation, this would use NLP to analyze the input
        emotional_stimuli = self._analyze_input_sentiment(user_input)
        
        # Update emotional state
        self.emotion_core.update_from_stimuli(emotional_stimuli)
        
        # Generate response
        response = self.behavior.generate_response(user_input, interaction_type)
        
        return response
    
    def _analyze_input_sentiment(self, text):
        """
        Analyze input text for emotional content
        
        This is a placeholder for actual sentiment analysis.
        In a real implementation, this would use NLP techniques.
        """
        # Simplified example
        stimuli = {}
        
        text_lower = text.lower()
        
        # Very basic keyword matching
        if any(word in text_lower for word in ["happy", "good", "great", "love", "like"]):
            stimuli["Joy"] = 0.1
            stimuli["Trust"] = 0.1
        
        if any(word in text_lower for word in ["sad", "sorry", "miss", "lonely"]):
            stimuli["Sadness"] = 0.1
        
        if any(word in text_lower for word in ["worried", "scared", "afraid", "nervous"]):
            stimuli["Fear"] = 0.1
        
        if any(word in text_lower for word in ["angry", "mad", "frustrated", "annoyed"]):
            stimuli["Anger"] = 0.1
            stimuli["Frustration"] = 0.1
        
        # Return neutral if no matches
        if not stimuli:
            stimuli = {
                "Surprise": 0.05,
                "Anticipation": 0.05
            }
        
        return stimuli
    
    def idle_update(self):
        """Update system state during idle periods"""
        # Apply emotional decay
        self.emotion_core.apply_decay()
        
        # Check if reflection is due
        if self.reflection.should_reflect():
            reflection = self.reflection.generate_reflection()
            
            # If Nyx should initiate based on state, return initiation message
            if self.behavior.should_initiate():
                comfort_request = self.behavior.generate_comfort_request()
                if comfort_request:
                    return comfort_request
                
                # Alternatively, share a reflection
                if reflection:
                    return f"I've been thinking... {reflection['summary']} {random.choice(reflection['insights'])}"
        
        return None
    
    def save_state(self, filename):
        """Save the current emotional and memory state to file"""
        state = {
            "emotions": self.emotion_core.get_emotional_state(),
            "memories": self.memory.get_all_memories(),
            "reflections": self.reflection.get_all_reflections(),
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction.isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filename):
        """Load emotional and memory state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Restore emotions
            for emotion, value in state["emotions"].items():
                self.emotion_core.emotions[emotion] = value
            
            # We would need more complex logic to fully restore all state components
            # This is a simplified version
            
            self.interaction_count = state["interaction_count"]
            self.last_interaction = datetime.datetime.fromisoformat(state["last_interaction"])
            
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False


# Usage example
if __name__ == "__main__":
    nyx = ProjectNyx()
    
    # Example interaction
    response = nyx.process_input("Hello Nyx, how are you feeling today?")
    print("Nyx:", response)
    
    # Update emotions directly
    nyx.emotion_core.update_emotion("Joy", 0.3)
    nyx.emotion_core.update_emotion("Trust", 0.2)
    
    # Process another input
    response = nyx.process_input("I've been thinking about you.")
    print("Nyx:", response)
    
    # Save state
    nyx.save_state("nyx_emotional_state.json")
