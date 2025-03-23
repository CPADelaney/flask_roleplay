# nyx/streamer/gamer_girl.py

import os
import asyncio
import logging
import time
import json
import numpy as np
import librosa
import sounddevice as sd
import cv2
import pickle
import difflib
from typing import List, Dict, Any, Optional, Tuple, Union, Literal, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque, Counter
from pathlib import Path

from pydantic import BaseModel, Field

# Import OpenAI Agents SDK
from agents import (
    Agent, Runner, ModelSettings, trace, function_tool, 
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    set_default_openai_key, handoff, TResponseInputItem,
    RunContextWrapper, enable_verbose_stdout_logging, WebSearchTool
)

# For speech recognition
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("game_agents")

# Set OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY environment variable must be set")

###########################################
# Game Audio Processor
###########################################

class GameAudioProcessor:
    """System for processing game audio and detecting sound events"""
    
    def __init__(self, sample_rate=16000):
        """Initialize audio processor with sample rate"""
        self.sample_rate = sample_rate
        self.audio_buffer = deque(maxlen=int(sample_rate * 5))  # 5 seconds buffer
        self.is_capturing = False
        self.detected_sounds = []
        self.audio_features = {}
        
    def start_capture(self, device=None):
        """Start audio capture from the specified device"""
        self.is_capturing = True
        logger.info(f"Started audio capture with sample rate {self.sample_rate}Hz")
        # In a real implementation, this would initialize audio device capture
    
    def stop_capture(self):
        """Stop audio capture"""
        self.is_capturing = False
        logger.info("Stopped audio capture")
    
    def process_audio_block(self, audio_data=None):
        """
        Process a block of audio data
        
        Args:
            audio_data: Optional audio data to process (if None, uses buffer)
        
        Returns:
            Dictionary of audio analysis results
        """
        # In a real implementation, this would analyze audio features
        # and detect sound events
        
        if audio_data is None:
            # Use synthetic data for testing
            audio_data = np.random.randn(1600)  # 0.1s of audio at 16kHz
        
        # Add to buffer
        for sample in audio_data:
            self.audio_buffer.append(sample)
        
        # Extract basic audio features
        self.audio_features = {
            "rms_energy": np.sqrt(np.mean(np.square(audio_data))),
            "zero_crossing_rate": np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data),
            "timestamp": time.time()
        }
        
        # Return analysis
        return self.audio_features
    
    def get_audio_features(self):
        """Get the latest audio features"""
        return self.audio_features
    
    def get_detected_sounds(self):
        """Get recently detected sounds"""
        return self.detected_sounds

###########################################
# Enhanced Game Vision Module
###########################################

class GameKnowledgeBase:
    """Knowledge base for game recognition and understanding"""
    
    def __init__(self):
        """Initialize game knowledge base"""
        self.games = {}
        self.objects = {}
        self.actions = {}
        self.locations = {}
    
    def load_game_data(self, game_id):
        """Load data for a specific game"""
        # In a real implementation, this would load game-specific data
        logger.info(f"Loaded game data for {game_id}")
        return True

class EnhancedGameRecognitionSystem:
    """System for recognizing games from visual input"""
    
    def __init__(self, knowledge_base=None):
        """
        Initialize game recognition system
        
        Args:
            knowledge_base: Optional GameKnowledgeBase instance
        """
        self.knowledge_base = knowledge_base or GameKnowledgeBase()
        self.recognized_games = {}
    
    async def identify_game(self, frame):
        """
        Identify the game being played from a video frame
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Dictionary with game identification information
        """
        # In a real implementation, this would use computer vision to identify games
        # For demonstration, return a placeholder result
        return {
            "game_id": "game123",
            "game_name": "Example Game",
            "confidence": 0.85,
            "genre": ["Action", "RPG"],
            "timestamp": time.time()
        }
    
    async def detect_objects(self, frame):
        """
        Detect game objects in a video frame
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            List of detected objects with bounding boxes
        """
        # In a real implementation, this would use object detection
        # For demonstration, return placeholder results
        return [
            {
                "class": "character",
                "name": "Player Character",
                "bbox": [100, 100, 200, 300],
                "confidence": 0.92
            },
            {
                "class": "item",
                "name": "Weapon",
                "bbox": [300, 250, 350, 300],
                "confidence": 0.85
            }
        ]

class EnhancedSpatialMemory:
    """Spatial memory system for game environments"""
    
    def __init__(self):
        """Initialize spatial memory system"""
        self.locations = {}
        self.current_location = None
        self.location_history = deque(maxlen=10)
    
    async def identify_location(self, frame, game_id=None):
        """
        Identify the current location in the game
        
        Args:
            frame: Video frame as numpy array
            game_id: Optional game ID for context
            
        Returns:
            Dictionary with location information
        """
        # In a real implementation, this would identify game locations
        # For demonstration, return a placeholder result
        location = {
            "id": "loc123",
            "name": "Forest Path",
            "zone": "Eastern Woods",
            "confidence": 0.78,
            "timestamp": time.time()
        }
        
        # Update tracking
        self.current_location = location
        self.location_history.append(location)
        
        return location

class SceneGraphAnalyzer:
    """Analyzes scene graphs for game understanding"""
    
    def __init__(self):
        """Initialize scene graph analyzer"""
        self.relationships = {}
    
    async def analyze_scene(self, objects, location):
        """
        Analyze relationships between objects in the scene
        
        Args:
            objects: List of detected objects
            location: Current location information
            
        Returns:
            Scene graph with object relationships
        """
        # In a real implementation, this would analyze object relationships
        # For demonstration, return a placeholder result
        return {
            "central_object": objects[0] if objects else None,
            "relationships": [
                {"subject": "Player Character", "relation": "holding", "object": "Weapon"}
            ],
            "scene_context": f"Player in {location.get('name', 'unknown location')}",
            "timestamp": time.time()
        }

class GameActionRecognition:
    """Recognizes player and NPC actions in games"""
    
    def __init__(self):
        """Initialize action recognition system"""
        self.recent_actions = deque(maxlen=5)
    
    async def detect_action(self, frames, audio_data=None):
        """
        Detect actions happening in a sequence of frames
        
        Args:
            frames: List of recent video frames
            audio_data: Optional audio data for multimodal detection
            
        Returns:
            Dictionary with action information
        """
        # In a real implementation, this would analyze actions using computer vision
        # For demonstration, return a placeholder result
        action = {
            "name": "Combat",
            "confidence": 0.82,
            "duration": 2.5,
            "involves_player": True,
            "timestamp": time.time()
        }
        
        # Update tracking
        self.recent_actions.append(action)
        
        return action

class RealTimeGameProcessor:
    """Real-time processor for game video streams"""
    
    def __init__(self, game_system=None, input_source=0, processing_fps=30):
        """
        Initialize real-time game processor
        
        Args:
            game_system: Game analysis system
            input_source: Video input source
            processing_fps: Target processing frame rate
        """
        self.game_system = game_system
        self.input_source = input_source
        self.processing_fps = processing_fps
        self.is_processing = False
        self.frame_count = 0
    
    def start_processing(self):
        """Start real-time processing"""
        self.is_processing = True
        logger.info(f"Started real-time game processing at {self.processing_fps} FPS")
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_processing = False
        logger.info("Stopped real-time game processing")

###########################################
# Hormone System
###########################################

class HormoneSystem:
    """System for simulating hormonal influences on emotions and behavior"""
    
    def __init__(self):
        """Initialize hormone system with default values"""
        # Initialize base hormones
        self.hormones = {
            "dopamine": {
                "value": 0.5,  # 0.0-1.0 scale
                "decay_rate": 0.05,
                "description": "Reward and pleasure hormone",
                "cycle_phase": "normal",
                "evolution_history": []
            },
            "serotonin": {
                "value": 0.5,
                "decay_rate": 0.03,
                "description": "Mood stabilizer and wellbeing hormone",
                "cycle_phase": "normal",
                "evolution_history": []
            },
            "cortisol": {
                "value": 0.3,
                "decay_rate": 0.08,
                "description": "Stress hormone",
                "cycle_phase": "normal",
                "evolution_history": []
            },
            "oxytocin": {
                "value": 0.4,
                "decay_rate": 0.04,
                "description": "Social bonding and trust hormone",
                "cycle_phase": "normal",
                "evolution_history": []
            },
            "adrenaline": {
                "value": 0.2,
                "decay_rate": 0.1,
                "description": "Fight-or-flight response hormone",
                "cycle_phase": "normal",
                "evolution_history": []
            }
        }
        
        # Environmental factors that influence hormones
        self.environmental_factors = {
            "time_of_day": 0.5,  # 0.0-1.0 representing 24 hour cycle
            "interaction_quality": 0.5,  # Quality of recent interactions
            "user_familiarity": 0.0,  # Familiarity with current user
            "session_duration": 0.0  # Length of current session
        }
        
        # Last update timestamp
        self.last_update = time.time()
        
        logger.info("Hormone system initialized with default values")
    
    async def update_hormone_cycles(self, ctx):
        """
        Update hormone levels based on natural cycles and environmental factors
        
        Args:
            ctx: Execution context
            
        Returns:
            Dictionary of updated hormone values
        """
        current_time = time.time()
        time_delta = current_time - self.last_update
        self.last_update = current_time
        
        # Update hormone values based on natural decay and environmental factors
        updates = {}
        
        for name, data in self.hormones.items():
            old_value = data["value"]
            
            # Natural decay toward baseline (0.5)
            baseline_pull = (0.5 - old_value) * data["decay_rate"] * time_delta
            
            # Environmental influences
            env_influence = self._calculate_environmental_influence(name)
            
            # Combined effect
            new_value = old_value + baseline_pull + env_influence
            new_value = max(0.0, min(1.0, new_value))  # Clamp to 0.0-1.0
            
            # Update hormone value
            self.hormones[name]["value"] = new_value
            
            # Record change
            if abs(new_value - old_value) > 0.01:  # Only record significant changes
                self.hormones[name]["evolution_history"].append({
                    "timestamp": current_time,
                    "old_value": old_value,
                    "new_value": new_value,
                    "change_reason": "cycle_update"
                })
            
            # Determine hormone phase
            if new_value > 0.7:
                self.hormones[name]["cycle_phase"] = "elevated"
            elif new_value < 0.3:
                self.hormones[name]["cycle_phase"] = "depleted"
            else:
                self.hormones[name]["cycle_phase"] = "normal"
            
            updates[name] = {
                "old_value": old_value,
                "new_value": new_value,
                "cycle_phase": self.hormones[name]["cycle_phase"]
            }
        
        return {
            "updated": True,
            "hormone_updates": updates,
            "environmental_factors": self.environmental_factors
        }
    
    def _calculate_environmental_influence(self, hormone_name):
        """
        Calculate environmental factor influence on a specific hormone
        
        Args:
            hormone_name: Name of the hormone
            
        Returns:
            Environmental influence value
        """
        influence = 0.0
        
        # Different hormones are influenced by different factors
        if hormone_name == "dopamine":
            # Dopamine influenced by interaction quality and time of day
            influence += self.environmental_factors["interaction_quality"] * 0.02
            
            # Higher in morning, lower at night
            time_influence = math.sin(self.environmental_factors["time_of_day"] * 2 * math.pi) * 0.01
            influence += time_influence
            
        elif hormone_name == "serotonin":
            # Serotonin influenced by time of day and familiarity
            # Higher during daylight hours
            time_influence = math.sin(self.environmental_factors["time_of_day"] * 2 * math.pi) * 0.015
            influence += time_influence
            
            # Higher with familiar users
            influence += self.environmental_factors["user_familiarity"] * 0.01
            
        elif hormone_name == "cortisol":
            # Cortisol influenced by session duration and time of day
            # Higher in morning, lower at night
            time_influence = math.cos(self.environmental_factors["time_of_day"] * 2 * math.pi) * 0.02
            influence += time_influence
            
            # Increases with very long sessions
            if self.environmental_factors["session_duration"] > 0.7:
                influence += (self.environmental_factors["session_duration"] - 0.7) * 0.05
            
        elif hormone_name == "oxytocin":
            # Oxytocin influenced by user familiarity and interaction quality
            influence += self.environmental_factors["user_familiarity"] * 0.02
            
            if self.environmental_factors["interaction_quality"] > 0.6:
                influence += (self.environmental_factors["interaction_quality"] - 0.6) * 0.04
            
        elif hormone_name == "adrenaline":
            # Adrenaline influenced by interaction quality (negatively)
            if self.environmental_factors["interaction_quality"] < 0.4:
                influence += (0.4 - self.environmental_factors["interaction_quality"]) * 0.05
            
            # Less adrenaline with familiar users
            influence -= self.environmental_factors["user_familiarity"] * 0.01
        
        return influence
    
    def update_hormone(self, hormone_name, value_change, reason="manual_update"):
        """
        Update a specific hormone level
        
        Args:
            hormone_name: Name of the hormone to update
            value_change: Amount to change the hormone value
            reason: Reason for the change
            
        Returns:
            Dictionary with update information
        """
        if hormone_name not in self.hormones:
            return {
                "updated": False,
                "error": f"Hormone {hormone_name} not found"
            }
        
        old_value = self.hormones[hormone_name]["value"]
        new_value = max(0.0, min(1.0, old_value + value_change))  # Clamp to 0.0-1.0
        
        # Update hormone value
        self.hormones[hormone_name]["value"] = new_value
        
        # Record change
        self.hormones[hormone_name]["evolution_history"].append({
            "timestamp": time.time(),
            "old_value": old_value,
            "new_value": new_value,
            "change_reason": reason
        })
        
        # Determine hormone phase
        if new_value > 0.7:
            self.hormones[hormone_name]["cycle_phase"] = "elevated"
        elif new_value < 0.3:
            self.hormones[hormone_name]["cycle_phase"] = "depleted"
        else:
            self.hormones[hormone_name]["cycle_phase"] = "normal"
        
        return {
            "updated": True,
            "hormone": hormone_name,
            "old_value": old_value,
            "new_value": new_value,
            "cycle_phase": self.hormones[hormone_name]["cycle_phase"]
        }
    
    def get_hormone_levels(self):
        """
        Get current hormone levels
        
        Returns:
            Dictionary of current hormone levels
        """
        return {name: data["value"] for name, data in self.hormones.items()}
    
    def get_emotional_state(self):
        """
        Get current emotional state based on hormone levels
        
        Returns:
            Dictionary with emotional state information
        """
        # Calculate emotional state based on hormone levels
        dopamine = self.hormones["dopamine"]["value"]
        serotonin = self.hormones["serotonin"]["value"]
        cortisol = self.hormones["cortisol"]["value"]
        oxytocin = self.hormones["oxytocin"]["value"]
        adrenaline = self.hormones["adrenaline"]["value"]
        
        # Calculate core emotional dimensions
        valence = (dopamine * 0.3 + serotonin * 0.3 + oxytocin * 0.2) - (cortisol * 0.1 + adrenaline * 0.1)
        valence = max(-1.0, min(1.0, valence * 2 - 0.5))  # Rescale to -1.0 to 1.0
        
        arousal = adrenaline * 0.4 + cortisol * 0.3 + dopamine * 0.2 - serotonin * 0.1
        arousal = max(-1.0, min(1.0, arousal * 2 - 0.5))  # Rescale to -1.0 to 1.0
        
        dominance = (dopamine * 0.3 + oxytocin * 0.2) - (cortisol * 0.3 + adrenaline * 0.2)
        dominance = max(-1.0, min(1.0, dominance * 2))  # Rescale to -1.0 to 1.0
        
        # Map to primary and secondary emotions
        primary_emotion, primary_intensity = self._map_to_primary_emotion(valence, arousal, dominance)
        secondary_emotion, secondary_intensity = self._map_to_secondary_emotion(
            valence, arousal, dominance, primary_emotion
        )
        
        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "primary_emotion": primary_emotion,
            "primary_intensity": primary_intensity,
            "secondary_emotion": secondary_emotion,
            "secondary_intensity": secondary_intensity,
            "hormone_levels": self.get_hormone_levels()
        }
    
    def _map_to_primary_emotion(self, valence, arousal, dominance):
        """
        Map VAD dimensions to primary emotion
        
        Args:
            valence: Pleasure-displeasure dimension (-1.0 to 1.0)
            arousal: Activation-deactivation dimension (-1.0 to 1.0)
            dominance: Dominance-submissiveness dimension (-1.0 to 1.0)
            
        Returns:
            Tuple of (emotion_name, intensity)
        """
        # Simplified mapping of VAD space to emotions
        if valence > 0.3:
            if arousal > 0.3:
                if dominance > 0.3:
                    return "Joy", min(1.0, (valence + arousal + dominance) / 2)
                else:
                    return "Trust", min(1.0, (valence + arousal) / 1.5)
            else:
                if dominance > 0.3:
                    return "Anticipation", min(1.0, (valence + dominance) / 1.5)
                else:
                    return "Serenity", min(1.0, valence / 0.7)
        else:
            if arousal > 0.3:
                if dominance > 0.3:
                    return "Anger", min(1.0, (arousal + dominance - valence) / 2)
                else:
                    return "Fear", min(1.0, (arousal - valence - dominance) / 1.5)
            else:
                if dominance > 0.3:
                    return "Disgust", min(1.0, (dominance - valence) / 1.5)
                else:
                    return "Sadness", min(1.0, (-valence) / 0.7)
        
        # Default
        return "Neutral", 0.5
    
    def _map_to_secondary_emotion(self, valence, arousal, dominance, primary_emotion):
        """
        Map VAD dimensions to secondary emotion (different from primary)
        
        Args:
            valence: Pleasure-displeasure dimension (-1.0 to 1.0)
            arousal: Activation-deactivation dimension (-1.0 to 1.0)
            dominance: Dominance-submissiveness dimension (-1.0 to 1.0)
            primary_emotion: Name of the primary emotion (to avoid duplication)
            
        Returns:
            Tuple of (emotion_name, intensity)
        """
        # Simplified secondary emotion mapping
        potential_emotions = []
        
        # Check for positive valence emotions
        if valence > 0:
            if primary_emotion != "Joy":
                potential_emotions.append(("Joy", valence * 0.7))
            if primary_emotion != "Trust":
                potential_emotions.append(("Trust", valence * 0.6))
            if primary_emotion != "Anticipation" and arousal > 0:
                potential_emotions.append(("Anticipation", valence * arousal * 0.8))
        
        # Check for negative valence emotions
        if valence < 0:
            if primary_emotion != "Sadness":
                potential_emotions.append(("Sadness", -valence * 0.7))
            if primary_emotion != "Fear" and arousal > 0:
                potential_emotions.append(("Fear", -valence * arousal * 0.8))
            if primary_emotion != "Disgust" and dominance > 0:
                potential_emotions.append(("Disgust", -valence * dominance * 0.7))
        
        # Add arousal-driven emotions
        if arousal > 0.3 and primary_emotion != "Surprise":
            potential_emotions.append(("Surprise", arousal * 0.6))
        
        if not potential_emotions:
            return "Neutral", 0.3
        
        # Sort by intensity and take the strongest
        potential_emotions.sort(key=lambda x: x[1], reverse=True)
        return potential_emotions[0]

###########################################
# Enhanced Multi-Modal Integration
###########################################

class EnhancedMultiModalIntegrator:
    """
    Enhanced integration between visual, audio, and speech modalities
    for more comprehensive game understanding and commentary.
    """
    
    def __init__(self, game_state):
        """Initialize with reference to game state"""
        self.game_state = game_state
        self.last_visual_update = 0
        self.last_audio_update = 0
        self.last_speech_update = 0
        self.combined_events = []
        self.audio_buffer = []
        self.speech_buffer = []
        self.visual_buffer = []
        
    async def process_frame(self, frame: np.ndarray, audio_data: np.ndarray = None):
        """Process a new frame with multi-modal integration"""
        current_time = time.time()
        
        # Add to visual buffer
        if frame is not None:
            self.visual_buffer.append((frame, current_time))
            # Keep buffer limited
            while len(self.visual_buffer) > 5:
                self.visual_buffer.pop(0)
        
        # Add to audio buffer
        if audio_data is not None:
            self.audio_buffer.append((audio_data, current_time))
            # Keep buffer limited
            while len(self.audio_buffer) > 10:
                self.audio_buffer.pop(0)
                
        # Look for multi-modal events
        events = await self._detect_combined_events()
        
        # Add significant events to game state
        for event in events:
            self.game_state.add_event(event["type"], event["data"])
            
        return events
    
    async def add_speech_event(self, speech_data: Dict[str, Any]):
        """Add a detected speech event"""
        current_time = time.time()
        self.speech_buffer.append((speech_data, current_time))
        
        # Keep buffer limited
        while len(self.speech_buffer) > 5:
            self.speech_buffer.pop(0)
            
        self.last_speech_update = current_time
    
    async def _detect_combined_events(self) -> List[Dict[str, Any]]:
        """Detect events by combining visual, audio, and speech cues"""
        events = []
        
        # Look for dialog events (speech + character on screen)
        if self.speech_buffer and self.visual_buffer:
            latest_speech, speech_time = self.speech_buffer[-1]
            latest_frame, frame_time = self.visual_buffer[-1]
            
            # If speech and frame are close in time
            if abs(speech_time - frame_time) < 2.0:
                # Check if character detection matches speaker
                if self.game_state.detected_objects:
                    for obj in self.game_state.detected_objects:
                        if obj.get("class") == "character" and latest_speech.get("speaker") == obj.get("name"):
                            # Found character speaking!
                            events.append({
                                "type": "character_dialog",
                                "data": {
                                    "character": obj.get("name"),
                                    "text": latest_speech.get("text"),
                                    "position": obj.get("bbox"),
                                    "confidence": latest_speech.get("confidence", 0.8),
                                    "significance": 8.0
                                }
                            })
        
        # Look for gameplay events (visual change + sound effect)
        if len(self.visual_buffer) >= 2 and len(self.audio_buffer) >= 1:
            # Basic change detection between frames
            if len(self.visual_buffer) >= 2:
                prev_frame, prev_time = self.visual_buffer[-2]
                curr_frame, curr_time = self.visual_buffer[-1]
                
                # Simple frame difference to detect visual changes
                if isinstance(prev_frame, np.ndarray) and isinstance(curr_frame, np.ndarray):
                    try:
                        # Convert to grayscale for simpler comparison
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate difference
                        diff = cv2.absdiff(prev_gray, curr_gray)
                        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                        change_percent = np.count_nonzero(thresh) / thresh.size
                        
                        # If significant visual change detected
                        if change_percent > 0.2:
                            # Check for accompanying audio
                            for audio_data, audio_time in self.audio_buffer:
                                # If audio is close in time to the visual change
                                if abs(audio_time - curr_time) < 0.5:
                                    # Calculate audio energy as a basic indicator
                                    audio_energy = np.mean(np.abs(audio_data))
                                    
                                    if audio_energy > 0.1:  # Significant audio
                                        events.append({
                                            "type": "gameplay_event",
                                            "data": {
                                                "visual_change": change_percent,
                                                "audio_energy": float(audio_energy),
                                                "location": self.game_state.current_location.get("name") if self.game_state.current_location else "unknown",
                                                "timestamp": curr_time,
                                                "significance": min(9.0, 5.0 + (change_percent * 10) + (audio_energy * 10))
                                            }
                                        })
                                        break
                    except Exception as e:
                        logger.error(f"Error in frame comparison: {e}")
        
        # Look for quest-related events (UI change + dialog)
        if self.speech_buffer and self.game_state.player_status:
            latest_speech = self.speech_buffer[-1][0]
            
            # Check for quest-related keywords in speech
            quest_keywords = ["quest", "mission", "objective", "task", "goal"]
            if any(keyword in latest_speech.get("text", "").lower() for keyword in quest_keywords):
                events.append({
                    "type": "quest_update",
                    "data": {
                        "text": latest_speech.get("text"),
                        "speaker": latest_speech.get("speaker"),
                        "current_objectives": self.game_state.player_status.get("objectives", []),
                        "significance": 7.5
                    }
                })
        
        return events

###########################################
# Context and State Management
###########################################

@dataclass
class GameState:
    """Game state information shared across agents"""
    # Game identification
    game_id: Optional[str] = None
    game_name: Optional[str] = None
    game_genre: Optional[str] = None
    game_mechanics: List[str] = field(default_factory=list)
    
    # Current frame and analysis
    current_frame: Optional[np.ndarray] = None
    frame_timestamp: Optional[float] = None
    
    # Audio analysis
    current_audio: Optional[np.ndarray] = None
    audio_features: Optional[Dict[str, Any]] = None
    detected_sounds: List[Dict[str, Any]] = field(default_factory=list)
    
    # Speech recognition
    transcribed_speech: deque = field(default_factory=lambda: deque(maxlen=10))
    dialog_history: List[Dict[str, Any]] = field(default_factory=list)
    character_voices: Dict[str, Any] = field(default_factory=dict)
    
    # Vision analysis results
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    current_location: Optional[Dict[str, Any]] = None
    detected_action: Optional[Dict[str, Any]] = None
    
    # Game context
    player_status: Dict[str, Any] = field(default_factory=dict)
    game_progress: Dict[str, Any] = field(default_factory=dict)
    quest_info: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-game knowledge
    similar_games: List[Dict[str, Any]] = field(default_factory=list)
    transferred_insights: List[Dict[str, Any]] = field(default_factory=list)
    mechanics_mapping: Dict[str, List[str]] = field(default_factory=dict)
    
    # Recent events for commentary
    recent_events: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Audience questions
    pending_questions: deque = field(default_factory=lambda: deque(maxlen=10))
    answered_questions: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Session data
    session_start_time: Optional[datetime] = None
    session_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Memory integration
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    
    # System state
    last_commentary_time: float = 0
    frame_count: int = 0
    
    def add_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Add a new event to the recent events queue"""
        self.recent_events.append({
            "type": event_type,
            "data": event_data,
            "timestamp": time.time()
        })
    
    def add_transcribed_speech(self, text: str, confidence: float, speaker: Optional[str] = None) -> None:
        """Add transcribed speech to the history"""
        speech_data = {
            "text": text,
            "confidence": confidence,
            "speaker": speaker,
            "timestamp": time.time()
        }
        self.transcribed_speech.append(speech_data)
        self.dialog_history.append(speech_data)
        
        # Also add as an event for commentary
        self.add_event("speech_transcribed", speech_data)
    
    def add_similar_game(self, game_name: str, similarity_score: float, similar_mechanics: List[str]) -> None:
        """Add a similar game to the knowledge base"""
        game_data = {
            "name": game_name,
            "similarity_score": similarity_score,
            "similar_mechanics": similar_mechanics,
            "added_at": time.time()
        }
        
        # Check if already exists and update if so
        for i, game in enumerate(self.similar_games):
            if game["name"] == game_name:
                self.similar_games[i] = game_data
                return
        
        # Otherwise add new
        self.similar_games.append(game_data)
    
    def add_transferred_insight(self, 
                               source_game: str, 
                               insight_type: str, 
                               insight_content: str,
                               relevance_score: float) -> None:
        """Add a transferred insight from another game"""
        insight_data = {
            "source_game": source_game,
            "type": insight_type,
            "content": insight_content,
            "relevance_score": relevance_score,
            "added_at": time.time()
        }
        self.transferred_insights.append(insight_data)
        
        # Also add as an event for commentary
        self.add_event("cross_game_insight", insight_data)
    
    def add_question(self, user_id: str, username: str, question: str) -> None:
        """Add an audience question to the queue"""
        self.pending_questions.append({
            "user_id": user_id,
            "username": username,
            "question": question,
            "timestamp": time.time()
        })
    
    def get_next_question(self) -> Optional[Dict[str, Any]]:
        """Get the next question from the queue"""
        if self.pending_questions:
            return self.pending_questions.popleft()
        return None
    
    def add_answered_question(self, question: Dict[str, Any], answer: str, feedback: Dict[str, Any] = None) -> None:
        """Store an answered question with optional feedback"""
        self.answered_questions.append({
            **question,
            "answer": answer,
            "feedback": feedback or {},
            "answered_at": time.time()
        })
        
        # Update session stats
        if "questions_answered" not in self.session_stats:
            self.session_stats["questions_answered"] = 0
        self.session_stats["questions_answered"] += 1

###########################################
# Speech Recognition System
###########################################

class SpeechRecognitionSystem:
    """System for recognizing and processing in-game speech and dialog"""
    
    def __init__(self, model_size="base", language="en"):
        """
        Initialize speech recognition system
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            language: Primary language expected in the audio
        """
        self.model_size = model_size
        self.language = language
        
        # Initialize Whisper model
        logger.info(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully")
        
        # Audio buffer (stores a few seconds of recent audio)
        self.audio_buffer = np.array([])
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.buffer_duration = 5  # seconds
        self.buffer_size = self.sample_rate * self.buffer_duration
        
        # Processing state
        self.is_processing = False
        self.last_processed_time = 0
        self.processing_interval = 2  # seconds
        
        # Known character voices
        self.character_voices = {}
        
        # Dialog history
        self.dialog_history = []
        
        # Transcription settings
        self.min_speech_duration = 0.5  # seconds
        self.min_speech_confidence = 0.6
        
        # Voice activity detection
        self.vad_threshold = 0.02  # Energy threshold for voice detection
    
    def start_capture(self, device=None):
        """Start audio capture for speech recognition"""
        logger.info(f"Started speech recognition audio capture with sample rate {self.sample_rate}Hz")
        # In a real implementation, this would initialize audio capture
    
    def stop_capture(self):
        """Stop audio capture"""
        logger.info("Stopped speech recognition audio capture")
    
    def add_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Add a chunk of audio data to the buffer
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate of the audio (will be resampled if necessary)
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.sample_rate
            )
        
        # Add to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_data)
        
        # Keep buffer at maximum size
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
    
    def _detect_voice_activity(self, audio: np.ndarray) -> bool:
        """
        Detect if there is voice activity in the audio segment
        
        Args:
            audio: Audio samples to check
            
        Returns:
            True if voice activity detected, False otherwise
        """
        # Calculate energy
        energy = np.mean(np.abs(audio))
        
        # Check if energy is above threshold
        return energy > self.vad_threshold
    
    async def process_audio(self) -> Optional[Dict[str, Any]]:
        """
        Process buffered audio and transcribe any speech
        
        Returns:
            Dictionary with transcription results if speech was detected, None otherwise
        """
        if self.is_processing or len(self.audio_buffer) < self.sample_rate * self.min_speech_duration:
            return None
        
        # Check if enough time has passed since last processing
        current_time = time.time()
        if current_time - self.last_processed_time < self.processing_interval:
            return None
        
        # Check for voice activity
        if not self._detect_voice_activity(self.audio_buffer):
            return None
        
        self.is_processing = True
        self.last_processed_time = current_time
        
        try:
            # Start with non-verbose processing
            result = await asyncio.to_thread(
                self.model.transcribe, 
                self.audio_buffer,
                language=self.language,
                fp16=False  # Use FP32 for compatibility
            )
            
            # Check if we got a valid transcription
            if not result or not result["text"] or len(result["text"].strip()) < 3:
                return None
            
            transcription = {
                "text": result["text"].strip(),
                "language": result.get("language", self.language),
                "segments": result.get("segments", []),
                "confidence": float(np.mean([s.get("confidence", 0) for s in result.get("segments", [])])),
                "timestamp": current_time
            }
            
            # Try to identify speaker (in a real implementation, this would use voice recognition)
            transcription["speaker"] = self._identify_speaker(self.audio_buffer, transcription["text"])
            
            # Add to dialog history
            self.dialog_history.append(transcription)
            
            # Only return if confidence is high enough
            if transcription["confidence"] >= self.min_speech_confidence:
                return transcription
            
            return None
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
        
        finally:
            self.is_processing = False
    
    def _identify_speaker(self, audio: np.ndarray, text: str) -> Optional[str]:
        """
        Identify the speaker based on voice characteristics and context
        
        Args:
            audio: Audio samples of the speech
            text: Transcribed text (for contextual hints)
            
        Returns:
            Speaker name if identified, None otherwise
        """
        # In a real implementation, this would use speaker diarization or voice identification
        # For this example, we'll use a simplified approach
        
        # Check if we have any character voices to compare with
        if not self.character_voices:
            return None
        
        # For demonstration, we'll randomly assign a speaker with 50% probability
        if np.random.random() < 0.5:
            return np.random.choice(list(self.character_voices.keys()))
        
        return None
    
    def add_character_voice(self, character_name: str, voice_samples: List[np.ndarray]):
        """
        Add a character's voice samples for identification
        
        Args:
            character_name: Name of the character
            voice_samples: List of audio samples of the character speaking
        """
        # In a real implementation, this would extract voice features
        self.character_voices[character_name] = {
            "samples": voice_samples,
            "added_at": time.time()
        }
        
        logger.info(f"Added voice profile for character: {character_name}")
    
    def get_recent_dialog(self, max_lines: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent dialog lines
        
        Args:
            max_lines: Maximum number of dialog lines to return
            
        Returns:
            List of recent dialog entries
        """
        return self.dialog_history[-max_lines:] if self.dialog_history else []

###########################################
# Cross-Game Knowledge System
###########################################

class GameMechanic(BaseModel):
    """Model for game mechanics"""
    name: str
    description: str
    examples: List[str] = Field(default_factory=list)
    games: List[str] = Field(default_factory=list)
    similar_mechanics: List[str] = Field(default_factory=list)

class CrossGameInsight(BaseModel):
    """Model for insights that can be transferred between games"""
    source_game: str
    target_game: str
    mechanic: str
    insight: str
    relevance: float = Field(ge=0.0, le=1.0)
    context: Optional[str] = None
    created_at: float = Field(default_factory=time.time)

class CrossGameKnowledgeSystem:
    """System for maintaining and transferring knowledge between similar games"""
    
    def __init__(self, data_dir="cross_game_data"):
        """
        Initialize cross-game knowledge system
        
        Args:
            data_dir: Directory to store knowledge data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Game knowledge base
        self.games_db = {}  # game_name -> game data
        self.mechanics_db = {}  # mechanic_name -> mechanic data
        self.insights_db = []  # list of cross-game insights
        
        # Similarity matrices
        self.game_similarity_matrix = {}  # game_name -> {other_game: similarity_score}
        self.mechanic_similarity_matrix = {}  # mechanic_name -> {other_mechanic: similarity_score}
        
        # Load existing knowledge
        self._load_knowledge()
        
        logger.info(f"Initialized cross-game knowledge system with {len(self.games_db)} games and {len(self.mechanics_db)} mechanics")
    
    def _load_knowledge(self):
        """Load knowledge bases from disk"""
        try:
            # Load games database
            games_path = self.data_dir / "games_db.json"
            if games_path.exists():
                with open(games_path, 'r') as f:
                    self.games_db = json.load(f)
            
            # Load mechanics database
            mechanics_path = self.data_dir / "mechanics_db.json"
            if mechanics_path.exists():
                with open(mechanics_path, 'r') as f:
                    mechanics_data = json.load(f)
                    # Convert to GameMechanic objects
                    self.mechanics_db = {name: GameMechanic(**data) for name, data in mechanics_data.items()}
            
            # Load insights database
            insights_path = self.data_dir / "insights_db.json"
            if insights_path.exists():
                with open(insights_path, 'r') as f:
                    insights_data = json.load(f)
                    # Convert to CrossGameInsight objects
                    self.insights_db = [CrossGameInsight(**data) for data in insights_data]
            
            # Load similarity matrices
            game_sim_path = self.data_dir / "game_similarity.json"
            if game_sim_path.exists():
                with open(game_sim_path, 'r') as f:
                    self.game_similarity_matrix = json.load(f)
            
            mechanic_sim_path = self.data_dir / "mechanic_similarity.json"
            if mechanic_sim_path.exists():
                with open(mechanic_sim_path, 'r') as f:
                    self.mechanic_similarity_matrix = json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading cross-game knowledge: {e}")
    
    def save_knowledge(self):
        """Save knowledge bases to disk"""
        try:
            # Save games database
            with open(self.data_dir / "games_db.json", 'w') as f:
                json.dump(self.games_db, f, indent=2)
            
            # Save mechanics database
            mechanics_data = {name: mechanic.dict() for name, mechanic in self.mechanics_db.items()}
            with open(self.data_dir / "mechanics_db.json", 'w') as f:
                json.dump(mechanics_data, f, indent=2)
            
            # Save insights database
            insights_data = [insight.dict() for insight in self.insights_db]
            with open(self.data_dir / "insights_db.json", 'w') as f:
                json.dump(insights_data, f, indent=2)
            
            # Save similarity matrices
            with open(self.data_dir / "game_similarity.json", 'w') as f:
                json.dump(self.game_similarity_matrix, f, indent=2)
                
            with open(self.data_dir / "mechanic_similarity.json", 'w') as f:
                json.dump(self.mechanic_similarity_matrix, f, indent=2)
                
            logger.info("Saved cross-game knowledge to disk")
            
        except Exception as e:
            logger.error(f"Error saving cross-game knowledge: {e}")
    
    def add_game(self, 
                 name: str, 
                 genre: List[str], 
                 mechanics: List[str], 
                 description: str, 
                 release_year: Optional[int] = None):
        """
        Add a game to the knowledge base
        
        Args:
            name: Game name
            genre: List of genres
            mechanics: List of game mechanics
            description: Game description
            release_year: Year the game was released
        """
        # Create game entry
        game_data = {
            "name": name,
            "genre": genre,
            "mechanics": mechanics,
            "description": description,
            "release_year": release_year,
            "added_at": time.time()
        }
        
        # Add to database
        self.games_db[name] = game_data
        
        # Update mechanic references
        for mechanic_name in mechanics:
            if mechanic_name in self.mechanics_db:
                # Add game to existing mechanic
                if name not in self.mechanics_db[mechanic_name].games:
                    mechanic = self.mechanics_db[mechanic_name]
                    mechanic.games.append(name)
                    self.mechanics_db[mechanic_name] = mechanic
        
        # Update similarity matrix
        self._update_game_similarities(name)
        
        logger.info(f"Added game to knowledge base: {name}")
        
        # Save updated knowledge
        self.save_knowledge()
        
        return game_data
    
    def add_mechanic(self, 
                    name: str, 
                    description: str, 
                    examples: List[str], 
                    games: List[str] = None):
        """
        Add a game mechanic to the knowledge base
        
        Args:
            name: Mechanic name
            description: Description of the mechanic
            examples: List of examples of the mechanic in action
            games: List of games that use this mechanic
        """
        # Create mechanic entry
        mechanic = GameMechanic(
            name=name,
            description=description,
            examples=examples,
            games=games or []
        )
        
        # Add to database
        self.mechanics_db[name] = mechanic
        
        # Update game references
        for game_name in games or []:
            if game_name in self.games_db:
                if name not in self.games_db[game_name]["mechanics"]:
                    self.games_db[game_name]["mechanics"].append(name)
        
        # Update similarity matrix
        self._update_mechanic_similarities(name)
        
        logger.info(f"Added mechanic to knowledge base: {name}")
        
        # Save updated knowledge
        self.save_knowledge()
        
        return mechanic
    
    def add_insight(self, 
                   source_game: str, 
                   target_game: str, 
                   mechanic: str, 
                   insight: str, 
                   relevance: float, 
                   context: Optional[str] = None):
        """
        Add an insight that can be transferred between games
        
        Args:
            source_game: Game the insight comes from
            target_game: Game the insight applies to
            mechanic: Game mechanic the insight relates to
            insight: The actual insight text
            relevance: How relevant the insight is (0.0 to 1.0)
            context: Optional context for when the insight applies
        """
        # Create insight entry
        insight_obj = CrossGameInsight(
            source_game=source_game,
            target_game=target_game,
            mechanic=mechanic,
            insight=insight,
            relevance=relevance,
            context=context
        )
        
        # Add to database
        self.insights_db.append(insight_obj)
        
        logger.info(f"Added cross-game insight: {source_game} -> {target_game}")
        
        # Save updated knowledge
        self.save_knowledge()
        
        return insight_obj
    
    def _update_game_similarities(self, game_name: str):
        """
        Update similarity scores between this game and other games
        
        Args:
            game_name: Name of the game to update similarities for
        """
        if game_name not in self.games_db:
            return
        
        game_data = self.games_db[game_name]
        similarities = {}
        
        for other_name, other_data in self.games_db.items():
            if other_name == game_name:
                continue
            
            # Calculate similarity score based on genres and mechanics
            genre_similarity = self._calculate_list_similarity(
                game_data.get("genre", []),
                other_data.get("genre", [])
            )
            
            mechanic_similarity = self._calculate_list_similarity(
                game_data.get("mechanics", []),
                other_data.get("mechanics", [])
            )
            
            # Weighted combination (mechanics matter more than genre)
            similarity = 0.3 * genre_similarity + 0.7 * mechanic_similarity
            
            similarities[other_name] = similarity
        
        # Update similarity matrix
        self.game_similarity_matrix[game_name] = similarities
    
    def _update_mechanic_similarities(self, mechanic_name: str):
        """
        Update similarity scores between this mechanic and other mechanics
        
        Args:
            mechanic_name: Name of the mechanic to update similarities for
        """
        if mechanic_name not in self.mechanics_db:
            return
        
        mechanic = self.mechanics_db[mechanic_name]
        similarities = {}
        
        for other_name, other_mechanic in self.mechanics_db.items():
            if other_name == mechanic_name:
                continue
            
            # Calculate text similarity between descriptions
            desc_similarity = self._calculate_text_similarity(
                mechanic.description,
                other_mechanic.description
            )
            
            # Calculate similarity based on games that use both mechanics
            games_overlap = len(set(mechanic.games) & set(other_mechanic.games))
            games_total = len(set(mechanic.games) | set(other_mechanic.games))
            games_similarity = games_overlap / max(1, games_total)
            
            # Weighted combination
            similarity = 0.5 * desc_similarity + 0.5 * games_similarity
            
            similarities[other_name] = similarity
            
            # Update similar mechanics lists if similarity is high enough
            if similarity > 0.7:
                if other_name not in mechanic.similar_mechanics:
                    mechanic.similar_mechanics.append(other_name)
                if mechanic_name not in other_mechanic.similar_mechanics:
                    other_mechanic.similar_mechanics.append(mechanic_name)
        
        # Update similarity matrix
        self.mechanic_similarity_matrix[mechanic_name] = similarities
    
    def _calculate_list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """
        Calculate similarity between two lists
        
        Args:
            list1: First list
            list2: Second list
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Jaccard similarity (size of intersection / size of union)
        set1 = set(list1)
        set2 = set(list2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / max(1, union)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Use SequenceMatcher for string similarity
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def get_similar_games(self, game_name: str, min_similarity: float = 0.4, max_games: int = 5) -> List[Dict[str, Any]]:
        """
        Get games similar to the specified game
        
        Args:
            game_name: Name of the game to find similarities for
            min_similarity: Minimum similarity score (0.0 to 1.0)
            max_games: Maximum number of games to return
            
        Returns:
            List of similar games with similarity scores
        """
        if game_name not in self.game_similarity_matrix:
            return []
        
        similarities = self.game_similarity_matrix[game_name]
        
        # Filter and sort by similarity
        similar_games = [
            {"name": other_game, "similarity": score}
            for other_game, score in similarities.items()
            if score >= min_similarity
        ]
        
        similar_games.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit to max_games
        similar_games = similar_games[:max_games]
        
        # Add mechanic overlaps
        if game_name in self.games_db:
            game_mechanics = set(self.games_db[game_name].get("mechanics", []))
            
            for game in similar_games:
                other_name = game["name"]
                if other_name in self.games_db:
                    other_mechanics = set(self.games_db[other_name].get("mechanics", []))
                    common_mechanics = list(game_mechanics & other_mechanics)
                    game["common_mechanics"] = common_mechanics
        
        return similar_games
    
    def get_applicable_insights(self, 
                               target_game: str, 
                               context: Optional[str] = None, 
                               min_relevance: float = 0.6, 
                               max_insights: int = 3) -> List[Dict[str, Any]]:
        """
        Get insights that can be applied to the target game
        
        Args:
            target_game: Name of the game to get insights for
            context: Optional context to filter insights
            min_relevance: Minimum relevance score (0.0 to 1.0)
            max_insights: Maximum number of insights to return
            
        Returns:
            List of applicable insights
        """
        # Direct insights (explicitly for this game)
        direct_insights = [
            insight.dict() 
            for insight in self.insights_db
            if insight.target_game == target_game and insight.relevance >= min_relevance
        ]
        
        # If we have context, filter by context relevance
        if context and direct_insights:
            for insight in direct_insights:
                if insight.get("context"):
                    # Calculate context similarity
                    context_similarity = self._calculate_text_similarity(context, insight["context"])
                    # Adjust relevance based on context match
                    insight["relevance"] *= (0.5 + 0.5 * context_similarity)
            
            # Re-filter by adjusted relevance
            direct_insights = [
                insight for insight in direct_insights 
                if insight["relevance"] >= min_relevance
            ]
        
        # Get similar games
        similar_games = self.get_similar_games(target_game)
        similar_game_names = [game["name"] for game in similar_games]
        
        # Indirect insights (from similar games)
        indirect_insights = []
        
        for source_game in similar_game_names:
            # Find insights from source game that might apply to target game
            game_insights = [
                insight.dict()
                for insight in self.insights_db
                if insight.source_game == source_game and 
                insight.target_game != target_game and  # Not already targeting our game
                insight.relevance >= min_relevance
            ]
            
            # Get similarity between source and target
            similarity = next(
                (game["similarity"] for game in similar_games if game["name"] == source_game),
                0.0
            )
            
            # Adjust relevance based on game similarity
            for insight in game_insights:
                insight["relevance"] *= similarity
                insight["original_target"] = insight["target_game"]
                insight["target_game"] = target_game  # Retarget to our game
                
                # If we have context, further adjust by context relevance
                if context and insight.get("context"):
                    context_similarity = self._calculate_text_similarity(context, insight["context"])
                    insight["relevance"] *= (0.5 + 0.5 * context_similarity)
            
            # Filter by adjusted relevance
            game_insights = [
                insight for insight in game_insights 
                if insight["relevance"] >= min_relevance
            ]
            
            indirect_insights.extend(game_insights)
        
        # Combine and sort by relevance
        all_insights = direct_insights + indirect_insights
        all_insights.sort(key=lambda x: x["relevance"], reverse=True)
        
        return all_insights[:max_insights]
    
    def seed_initial_knowledge(self):
        """Seed the knowledge base with initial game data"""
        # Add some common game mechanics
        mechanics = [
            {
                "name": "health_regeneration",
                "description": "Player health automatically regenerates over time",
                "examples": ["Resting to heal", "Regenerating health bar", "Time-based healing"]
            },
            {
                "name": "crafting",
                "description": "Combining items to create new ones",
                "examples": ["Crafting weapons", "Creating potions", "Building structures"]
            },
            {
                "name": "skill_tree",
                "description": "A branching progression system for character abilities",
                "examples": ["Talent trees", "Ability upgrades", "Specialization paths"]
            },
            {
                "name": "open_world",
                "description": "A large, continuous game world that can be freely explored",
                "examples": ["Open map exploration", "Non-linear progression", "Discoverable locations"]
            },
            {
                "name": "fast_travel",
                "description": "The ability to quickly travel between discovered locations",
                "examples": ["Travel points", "Teleportation", "Quick map movement"]
            }
        ]
        
        # Add mechanics to database
        for mechanic_data in mechanics:
            self.add_mechanic(
                name=mechanic_data["name"],
                description=mechanic_data["description"],
                examples=mechanic_data["examples"]
            )
        
        # Add some games
        games = [
            {
                "name": "The Witcher 3",
                "genre": ["RPG", "Open World", "Action"],
                "mechanics": ["health_regeneration", "crafting", "skill_tree", "open_world", "fast_travel"],
                "description": "An action RPG set in a fantasy world, following the adventures of monster hunter Geralt of Rivia.",
                "release_year": 2015
            },
            {
                "name": "Skyrim",
                "genre": ["RPG", "Open World", "Action"],
                "mechanics": ["health_regeneration", "crafting", "skill_tree", "open_world", "fast_travel"],
                "description": "An open world action RPG set in the province of Skyrim, where the player character can explore freely and develop their character.",
                "release_year": 2011
            },
            {
                "name": "Elden Ring",
                "genre": ["Action", "RPG", "Open World"],
                "mechanics": ["skill_tree", "crafting", "open_world", "fast_travel"],
                "description": "An action RPG set in a fantasy world created by Hidetaka Miyazaki and George R. R. Martin.",
                "release_year": 2022
            },
            {
                "name": "God of War (2018)",
                "genre": ["Action", "Adventure"],
                "mechanics": ["skill_tree", "crafting"],
                "description": "An action-adventure game following Kratos and his son Atreus on a journey through Norse realms.",
                "release_year": 2018
            },
            {
                "name": "Horizon Zero Dawn",
                "genre": ["Action", "RPG", "Open World"],
                "mechanics": ["crafting", "skill_tree", "open_world", "fast_travel"],
                "description": "An open world action RPG set in a post-apocalyptic world dominated by robotic creatures.",
                "release_year": 2017
            }
        ]
        
        # Add games to database
        for game_data in games:
            self.add_game(
                name=game_data["name"],
                genre=game_data["genre"],
                mechanics=game_data["mechanics"],
                description=game_data["description"],
                release_year=game_data["release_year"]
            )
        
        # Add some cross-game insights
        insights = [
            {
                "source_game": "The Witcher 3",
                "target_game": "Elden Ring",
                "mechanic": "open_world",
                "insight": "Unlike The Witcher 3's guided open world, Elden Ring requires more self-directed exploration with minimal guidance.",
                "relevance": 0.8,
                "context": "exploration"
            },
            {
                "source_game": "Skyrim",
                "target_game": "The Witcher 3",
                "mechanic": "skill_tree",
                "insight": "While Skyrim lets you develop any skill by using it, The Witcher 3 uses a more traditional point-based skill tree with mutagens for additional effects.",
                "relevance": 0.9,
                "context": "character building"
            },
            {
                "source_game": "God of War (2018)",
                "target_game": "Horizon Zero Dawn",
                "mechanic": "crafting",
                "insight": "Both games use crafting for equipment upgrades, but Horizon Zero Dawn emphasizes gathering resources from defeated machines.",
                "relevance": 0.7,
                "context": "combat preparation"
            }
        ]
        
        # Add insights to database
        for insight_data in insights:
            self.add_insight(
                source_game=insight_data["source_game"],
                target_game=insight_data["target_game"],
                mechanic=insight_data["mechanic"],
                insight=insight_data["insight"],
                relevance=insight_data["relevance"],
                context=insight_data["context"]
            )
        
        logger.info("Seeded cross-game knowledge base with initial data")
    
    def discover_patterns(self, games: List[str]) -> Dict[str, Any]:
        """
        Discover patterns and relationships across a set of games
        
        Args:
            games: List of game names to analyze
            
        Returns:
            Dictionary of discovered patterns and relationships
        """
        # Get all games data
        games_data = {}
        for game in games:
            if game in self.games_db:
                games_data[game] = self.games_db[game]
        
        if not games_data:
            return {"error": "No valid games found in knowledge base"}
        
        # Find common mechanics
        all_mechanics = {}
        for game, data in games_data.items():
            for mechanic in data.get("mechanics", []):
                if mechanic not in all_mechanics:
                    all_mechanics[mechanic] = []
                all_mechanics[mechanic].append(game)
        
        common_mechanics = {
            mechanic: game_list
            for mechanic, game_list in all_mechanics.items()
            if len(game_list) > 1
        }
        
        # Find common genres
        all_genres = {}
        for game, data in games_data.items():
            for genre in data.get("genre", []):
                if genre not in all_genres:
                    all_genres[genre] = []
                all_genres[genre].append(game)
        
        common_genres = {
            genre: game_list
            for genre, game_list in all_genres.items()
            if len(game_list) > 1
        }
        
        # Find innovative mechanics (only in one game)
        innovative_mechanics = {
            mechanic: game_list
            for mechanic, game_list in all_mechanics.items()
            if len(game_list) == 1
        }
        
        return {
            "common_mechanics": common_mechanics,
            "common_genres": common_genres,
            "innovative_mechanics": innovative_mechanics,
            "games_analyzed": len(games_data)
        }
    
    def generate_insight(self, 
                       source_game: str, 
                       target_game: str,
                       mechanic: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a new insight between two games
        
        Args:
            source_game: Source game for the insight
            target_game: Target game for the insight
            mechanic: Optional specific mechanic to focus on
            
        Returns:
            Generated insight
        """
        # Check if games exist
        if source_game not in self.games_db or target_game not in self.games_db:
            return {"error": "One or both games not found in knowledge base"}
        
        source_data = self.games_db[source_game]
        target_data = self.games_db[target_game]
        
        # Find common mechanics if not specified
        if not mechanic:
            source_mechanics = set(source_data.get("mechanics", []))
            target_mechanics = set(target_data.get("mechanics", []))
            common_mechanics = source_mechanics.intersection(target_mechanics)
            
            if not common_mechanics:
                return {"error": "No common mechanics found between games"}
            
            # Pick the first common mechanic
            mechanic = list(common_mechanics)[0]
        elif mechanic not in source_data.get("mechanics", []) or mechanic not in target_data.get("mechanics", []):
            return {"error": f"Mechanic {mechanic} not found in both games"}
        
        # Get mechanic details
        mechanic_data = self.mechanics_db.get(mechanic, None)
        if not mechanic_data:
            return {"error": f"Mechanic {mechanic} not found in knowledge base"}
        
        # Generate a basic insight
        # In a real system, this would use NLP to generate better insights
        insight = f"Both {source_game} and {target_game} use {mechanic}, but they implement it differently based on their game design philosophies."
        
        # Add to insights database
        insight_obj = self.add_insight(
            source_game=source_game,
            target_game=target_game,
            mechanic=mechanic,
            insight=insight,
            relevance=0.7
        )
        
        return {
            "source_game": source_game,
            "target_game": target_game,
            "mechanic": mechanic,
            "insight": insight,
            "relevance": 0.7,
            "id": str(time.time())
        }
    
    def consolidate_knowledge(self) -> Dict[str, Any]:
        """
        Consolidate knowledge to identify patterns and create new insights
        
        Returns:
            Results of the consolidation process
        """
        # In a real system, this would run advanced analytics
        # For demonstration, we'll create some simple insights
        
        # 1. Find most similar games
        all_similarities = []
        for game, similarities in self.game_similarity_matrix.items():
            for other_game, score in similarities.items():
                if score > 0.7:  # High similarity threshold
                    all_similarities.append((game, other_game, score))
        
        # Sort by similarity score
        all_similarities.sort(key=lambda x: x[2], reverse=True)
        
        # 2. Generate insights for top similar pairs
        new_insights = []
        for game1, game2, score in all_similarities[:5]:
            # Get common mechanics
            mechanics1 = set(self.games_db.get(game1, {}).get("mechanics", []))
            mechanics2 = set(self.games_db.get(game2, {}).get("mechanics", []))
            common = mechanics1.intersection(mechanics2)
            
            if common:
                mechanic = list(common)[0]
                result = self.generate_insight(game1, game2, mechanic)
                if "error" not in result:
                    new_insights.append(result)
        
        # 3. Identify emerging patterns
        # Get all mechanics that appear in at least 3 games
        popular_mechanics = []
        for mechanic, data in self.mechanics_db.items():
            if len(data.games) >= 3:
                popular_mechanics.append((mechanic, len(data.games)))
        
        popular_mechanics.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "new_insights": new_insights,
            "popular_mechanics": popular_mechanics[:5],
            "top_similar_games": [(g1, g2, round(score, 2)) for g1, g2, score in all_similarities[:5]]
        }

###########################################
# Game Analysis Functions
###########################################

async def identify_game(ctx: RunContextWrapper[GameState]) -> str:
    """
    Identify the game being played from the current frame
    
    Returns:
        Information about the identified game
    """
    game_state = ctx.context
    
    # Check if we already identified the game
    if game_state.game_id:
        return f"Game already identified as {game_state.game_name} (ID: {game_state.game_id})"
    
    # Check if we have a frame to analyze
    if game_state.current_frame is None:
        return "No frame available for game identification"
    
    # Create game recognition system if not available in context
    recognition_system = None
    if hasattr(ctx, "game_recognition"):
        recognition_system = ctx.game_recognition
    else:
        recognition_system = EnhancedGameRecognitionSystem()
    
    # Identify the game
    result = await recognition_system.identify_game(game_state.current_frame)
    
    if result and result.get("confidence", 0) > 0.7:
        # Update game state
        game_state.game_id = result.get("game_id")
        game_state.game_name = result.get("game_name")
        game_state.game_genre = result.get("genre", [])
        
        return f"Identified game as {game_state.game_name} (ID: {game_state.game_id}) with {result.get('confidence', 0):.2f} confidence"
    
    return "Could not identify game with sufficient confidence"

async def analyze_current_frame(ctx: RunContextWrapper[GameState]) -> str:
    """
    Analyze the current frame to extract objects, characters, and UI elements
    
    Returns:
        Analysis of the current frame
    """
    game_state = ctx.context
    
    # Check if we have a frame to analyze
    if game_state.current_frame is None:
        return "No frame available for analysis"
    
    # Create recognition system if not available in context
    recognition_system = None
    if hasattr(ctx, "game_recognition"):
        recognition_system = ctx.game_recognition
    else:
        recognition_system = EnhancedGameRecognitionSystem()
    
    # Detect objects
    objects = await recognition_system.detect_objects(game_state.current_frame)
    
    # Update game state
    game_state.detected_objects = objects
    
    # Format results
    if not objects:
        return "No notable objects detected in the current frame"
    
    object_descriptions = []
    for obj in objects:
        obj_class = obj.get("class", "object")
        obj_name = obj.get("name", "unknown")
        confidence = obj.get("confidence", 0)
        
        object_descriptions.append(f"{obj_name} ({obj_class}, {confidence:.2f} confidence)")
    
    return "Detected in current frame: " + ", ".join(object_descriptions)

async def get_player_location(ctx: RunContextWrapper[GameState]) -> str:
    """
    Determine the player's current location in the game
    
    Returns:
        Current location information
    """
    game_state = ctx.context
    
    # Check if we have a frame to analyze
    if game_state.current_frame is None:
        return "No frame available for location analysis"
    
    # Create spatial memory if not available in context
    spatial_memory = None
    if hasattr(ctx, "spatial_memory"):
        spatial_memory = ctx.spatial_memory
    else:
        spatial_memory = EnhancedSpatialMemory()
    
    # Identify location
    location = await spatial_memory.identify_location(
        game_state.current_frame,
        game_state.game_id
    )
    
    # Update game state
    game_state.current_location = location
    
    # Format results
    if not location:
        return "Could not identify current location"
    
    location_name = location.get("name", "unknown area")
    zone = location.get("zone", "unknown zone")
    confidence = location.get("confidence", 0)
    
    return f"Current location: {location_name} in {zone} ({confidence:.2f} confidence)"

async def detect_current_action(ctx: RunContextWrapper[GameState]) -> str:
    """
    Detect what action is currently happening in the game
    
    Returns:
        Current action information
    """
    game_state = ctx.context
    
    # Check if we have recent frames to analyze
    if game_state.current_frame is None:
        return "No frames available for action detection"
    
    # Get recent frames from buffer if available
    frames = [game_state.current_frame]  # Use at least the current frame
    
    # Create action recognition if not available in context
    action_recognition = None
    if hasattr(ctx, "action_recognition"):
        action_recognition = ctx.action_recognition
    else:
        action_recognition = GameActionRecognition()
    
    # Detect action
    action = await action_recognition.detect_action(
        frames,
        game_state.current_audio
    )
    
    # Update game state
    game_state.detected_action = action
    
    # Format results
    if not action:
        return "Could not detect current action"
    
    action_name = action.get("name", "unknown action")
    confidence = action.get("confidence", 0)
    player_involved = "involving the player" if action.get("involves_player", False) else "not involving the player"
    
    return f"Current action: {action_name} ({confidence:.2f} confidence, {player_involved})"

###########################################
# Learning Analysis System
###########################################

class GameSessionLearningManager:
    """
    System for analyzing and summarizing learnings from streaming sessions
    """
    
    def __init__(self, brain, streaming_core):
        """
        Initialize learning manager with references to brain and streaming system
        
        Args:
            brain: NyxBrain instance
            streaming_core: StreamingCore instance
        """
        self.brain = brain
        self.streaming_core = streaming_core
        self.learnings = []
        self.last_analysis_time = time.time()
        self.analysis_interval = 600  # 10 minutes
        self.learning_categories = {
            "game_mechanics": [],
            "storytelling": [],
            "audience_engagement": [],
            "cross_game_insights": [],
            "technical": []
        }
    
    async def analyze_session_learnings(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze what has been learned during a streaming session
        
        Args:
            session_data: Data from the current streaming session
            
        Returns:
            Learning analysis results
        """
        # Extract relevant data
        game_name = session_data.get("game_name", "Unknown Game")
        recent_events = session_data.get("recent_events", [])
        dialog_history = session_data.get("dialog_history", [])
        answered_questions = session_data.get("answered_questions", [])
        transferred_insights = session_data.get("transferred_insights", [])
        
        # Extract potential learnings from different sources
        new_learnings = []
        
        # From audience questions and answers
        for qa in answered_questions[-10:]:  # Focus on most recent
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            
            # Look for knowledge-rich answers
            if len(answer) > 100 and any(kw in question.lower() for kw in ["how", "why", "what", "explain"]):
                # This Q&A might contain learning
                category = self._categorize_learning(question, answer)
                
                # Create learning item
                learning = {
                    "text": f"Learned about {category} while explaining: {question[:50]}...",
                    "detail": answer,
                    "category": category,
                    "source": "audience_question",
                    "timestamp": qa.get("timestamp", time.time())
                }
                
                new_learnings.append(learning)
                self.learning_categories[category].append(learning)
        
        # From cross-game insights
        for insight in transferred_insights:
            # Create learning item from insight
            category = "cross_game_insights"
            
            learning = {
                "text": f"Gained insight comparing {game_name} to {insight.get('source_game', 'another game')}",
                "detail": insight.get("content", ""),
                "category": category,
                "source": "cross_game",
                "timestamp": insight.get("added_at", time.time())
            }
            
            new_learnings.append(learning)
            self.learning_categories[category].append(learning)
        
        # From dialog (lore and story learnings)
        story_content = []
        for dialog in dialog_history[-15:]:  # Focus on recent dialog
            if "speaker" in dialog and dialog.get("speaker") != "Unknown":
                # Named character dialog might contain lore
                story_content.append(dialog.get("text", ""))
        
        if story_content:
            # Combine related dialog into single learning
            story_text = " ".join(story_content[:5])  # Limit length
            
            learning = {
                "text": f"Learned story information in {game_name}",
                "detail": story_text,
                "category": "storytelling",
                "source": "dialog",
                "timestamp": time.time()
            }
            
            new_learnings.append(learning)
            self.learning_categories["storytelling"].append(learning)
        
        # Combine with existing learnings
        self.learnings.extend(new_learnings)
        
        # Return analysis results
        return {
            "new_learnings": len(new_learnings),
            "total_learnings": len(self.learnings),
            "categories": {k: len(v) for k, v in self.learning_categories.items()},
            "latest_learning": new_learnings[0]["text"] if new_learnings else None
        }
    
    def _categorize_learning(self, question: str, answer: str) -> str:
        """
        Categorize a learning based on question and answer content
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Learning category
        """
        # Simple keyword-based categorization
        text = (question + " " + answer).lower()
        
        # Game mechanics 
        if any(kw in text for kw in ["mechanics", "controls", "gameplay", "system", "how to", "abilities"]):
            return "game_mechanics"
        
        # Storytelling
        if any(kw in text for kw in ["story", "character", "plot", "lore", "world", "setting"]):
            return "storytelling"
        
        # Technical
        if any(kw in text for kw in ["engine", "graphics", "performance", "development", "technical"]):
            return "technical"
        
        # Audience engagement
        if any(kw in text for kw in ["audience", "stream", "viewer", "chat", "community"]):
            return "audience_engagement"
        
        # Default to game mechanics
        return "game_mechanics"
    
    async def generate_learning_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of what has been learned during the session
        
        Returns:
            Summary of learnings
        """
        # Check if we have enough learnings
        if len(self.learnings) < 2:
            return {
                "has_learnings": False,
                "summary": "Not enough learnings have been recorded yet to generate a summary."
            }
        
        # Get top categories
        category_counts = {k: len(v) for k, v in self.learning_categories.items()}
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Generate summary text
        summary_lines = [
            f"During this streaming session, I've learned {len(self.learnings)} things:"
        ]
        
        # Add top 3 categories
        for category, count in top_categories[:3]:
            if count > 0:
                # Get most recent learning from this category
                recent = self.learning_categories[category][-1]
                
                # Format category name for display
                display_category = category.replace("_", " ").title()
                
                summary_lines.append(f"• {display_category} ({count}): {recent['text']}")
        
        # Add note about cross-game insights if available
        if category_counts.get("cross_game_insights", 0) > 0:
            insights = self.learning_categories["cross_game_insights"]
            summary_lines.append(f"• Made {len(insights)} connections to other games")
        
        # Combine into final summary
        summary = "\n".join(summary_lines)
        
        return {
            "has_learnings": True,
            "summary": summary,
            "total_learnings": len(self.learnings),
            "categories": category_counts,
            "top_category": top_categories[0][0] if top_categories else None
        }
    
    async def assess_functionality_needs(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess what functionality might be needed based on learning patterns
        
        Args:
            summary_data: Summary data of learnings
            
        Returns:
            Assessment of functionality needs
        """
        functionality_needs = []
        explanation = []
        
        # Get category data
        categories = summary_data.get("categories", {})
        total_learnings = summary_data.get("total_learnings", 0)
        
        # Check for potential needs
        if total_learnings > 20:  # Significant learning activity
            if categories.get("cross_game_insights", 0) > 5:
                functionality_needs.append("cross_game_knowledge_expansion")
                explanation.append("High cross-game learning suggests expanding the cross-game knowledge system")
                
            if categories.get("storytelling", 0) > categories.get("game_mechanics", 0):
                functionality_needs.append("narrative_analysis")
                explanation.append("Focus on story suggests adding narrative analysis capabilities")
                
            if categories.get("audience_engagement", 0) > 5:
                functionality_needs.append("audience_interaction_enhancement")
                explanation.append("High audience engagement suggests enhancing interaction capabilities")
        
        return {
            "functionality_needs": functionality_needs,
            "explanation": explanation,
            "learning_volume": "high" if total_learnings > 20 else "medium" if total_learnings > 10 else "low"
        }
        
###########################################
# Enhanced Agent Tools
###########################################

@function_tool
async def analyze_speech(ctx: RunContextWrapper[GameState]) -> str:
    """
    Process and transcribe any speech or dialog from the game audio.
    
    Returns:
        Transcription of any detected speech, or a message indicating no speech was detected.
    """
    # Check if we have a speech recognition system in the context
    if not hasattr(ctx, "speech_recognition"):
        return "Speech recognition not available."
    
    speech_system = ctx.speech_recognition
    
    # Process current audio
    transcription = await speech_system.process_audio()
    
    if not transcription:
        # Check if we have any recent speech in the game state
        if ctx.context.transcribed_speech:
            latest_speech = ctx.context.transcribed_speech[-1]
            return f"No new speech detected. Most recent speech: \"{latest_speech['text']}\" (Speaker: {latest_speech['speaker'] or 'Unknown'})"
        else:
            return "No speech detected in the current audio."
    
    # Add to game state
    ctx.context.add_transcribed_speech(
        text=transcription["text"],
        confidence=transcription["confidence"],
        speaker=transcription.get("speaker")
    )
    
    # Format response
    if transcription.get("speaker"):
        return f"Transcribed speech from {transcription['speaker']}: \"{transcription['text']}\" (Confidence: {transcription['confidence']:.2f})"
    else:
        return f"Transcribed speech: \"{transcription['text']}\" (Confidence: {transcription['confidence']:.2f})"

@function_tool
async def get_recent_dialog(ctx: RunContextWrapper[GameState], max_lines: int = 5) -> str:
    """
    Get recent dialog lines transcribed from the game.
    
    Args:
        max_lines: Maximum number of dialog lines to return
        
    Returns:
        Recent dialog with speakers if identified.
    """
    # Check if we have any transcribed speech
    if not ctx.context.dialog_history:
        return "No dialog has been transcribed yet."
    
    # Get recent dialog
    recent = ctx.context.dialog_history[-max_lines:]
    
    # Format dialog
    lines = []
    for entry in recent:
        if entry.get("speaker"):
            lines.append(f"{entry['speaker']}: \"{entry['text']}\"")
        else:
            lines.append(f"Unknown: \"{entry['text']}\"")
    
    return "Recent dialog:\n" + "\n".join(lines)

@function_tool
async def find_similar_games(ctx: RunContextWrapper[GameState]) -> str:
    """
    Find games similar to the current game based on genre and mechanics.
    
    Returns:
        List of similar games with similarity explanations.
    """
    # Check if we have a cross-game knowledge system
    if not hasattr(ctx, "cross_game_knowledge"):
        return "Cross-game knowledge system not available."
    
    knowledge_system = ctx.cross_game_knowledge
    game_state = ctx.context
    
    # Check if we have a current game
    if not game_state.game_name:
        return "No current game identified."
    
    # Get similar games
    similar_games = knowledge_system.get_similar_games(game_state.game_name)
    
    if not similar_games:
        return f"No similar games found for {game_state.game_name}."
    
    # Update game state
    for game in similar_games:
        game_state.add_similar_game(
            game_name=game["name"],
            similarity_score=game["similarity"],
            similar_mechanics=game.get("common_mechanics", [])
        )
    
    # Format response
    lines = [f"Games similar to {game_state.game_name}:"]
    
    for game in similar_games:
        similarity_pct = int(game["similarity"] * 100)
        common_mechanics = game.get("common_mechanics", [])
        
        if common_mechanics:
            mechanics_str = ", ".join(common_mechanics)
            lines.append(f"• {game['name']} ({similarity_pct}% similar) - Shared mechanics: {mechanics_str}")
        else:
            lines.append(f"• {game['name']} ({similarity_pct}% similar)")
    
    return "\n".join(lines)

@function_tool
async def get_cross_game_insights(ctx: RunContextWrapper[GameState], context: Optional[str] = None) -> str:
    """
    Get insights from other games that apply to the current game situation.
    
    Args:
        context: Optional context for the current gameplay situation
        
    Returns:
        Applicable insights from other games.
    """
    # Check if we have a cross-game knowledge system
    if not hasattr(ctx, "cross_game_knowledge"):
        return "Cross-game knowledge system not available."
    
    knowledge_system = ctx.cross_game_knowledge
    game_state = ctx.context
    
    # Check if we have a current game
    if not game_state.game_name:
        return "No current game identified."
    
    # Get applicable insights
    insights = knowledge_system.get_applicable_insights(
        target_game=game_state.game_name,
        context=context
    )
    
    if not insights:
        return f"No applicable insights found for {game_state.game_name}."
    
    # Update game state
    for insight in insights:
        game_state.add_transferred_insight(
            source_game=insight["source_game"],
            insight_type=insight["mechanic"],
            insight_content=insight["insight"],
            relevance_score=insight["relevance"]
        )
    
    # Format response
    lines = [f"Insights applicable to {game_state.game_name}:"]
    
    for insight in insights:
        relevance_pct = int(insight["relevance"] * 100)
        lines.append(f"• From {insight['source_game']} ({relevance_pct}% relevant): {insight['insight']}")
    
    return "\n".join(lines)

@function_tool
async def search_game_information(ctx: RunContextWrapper[GameState], search_query: str) -> str:
    """
    Search for additional information about the current game or gaming concepts.
    
    Args:
        search_query: The query to search for
        
    Returns:
        Search results about the game.
    """
    # Check if we have a current game
    game_state = ctx.context
    
    if not game_state.game_name:
        search_term = search_query
    else:
        # Add game name to the search query for context
        search_term = f"{game_state.game_name} {search_query}"
    
    # Use OpenAI Agents SDK WebSearchTool
    if hasattr(ctx, "web_search_tool"):
        search_tool = ctx.web_search_tool
        result = await search_tool.invoke(search_term)
        return f"Search results for '{search_term}':\n{result}"
    else:
        return f"Web search tool not available. Query was: '{search_term}'"

@function_tool
async def get_learning_summary(ctx: RunContextWrapper[GameState]) -> str:
    """
    Get a summary of what has been learned during the streaming session
    
    Returns:
        Summary of session learnings
    """
    game_state = ctx.context
    
    # Check if there's a learning manager available
    if not hasattr(ctx, "learning_manager"):
        return "Learning analysis system not available."
    
    learning_manager = ctx.learning_manager
    
    # Get session data
    session_data = {
        "game_name": game_state.game_name,
        "recent_events": list(game_state.recent_events),
        "dialog_history": game_state.dialog_history,
        "answered_questions": list(game_state.answered_questions),
        "transferred_insights": game_state.transferred_insights
    }
    
    # Generate learning summary
    try:
        summary_result = await learning_manager.generate_learning_summary()
        
        if summary_result.get("has_learnings", False):
            return summary_result["summary"]
        else:
            return "No significant learnings have been identified in this streaming session yet."
    except Exception as e:
        return f"Error generating learning summary: {e}"

###########################################
# Enhanced Audience Interaction
###########################################

class EnhancedAudienceInteraction:
    """
    Enhanced audience interaction system with improved question answering,
    personalization, and audience analytics.
    """
    
    def __init__(self, game_state: GameState):
        """Initialize with reference to game state"""
        self.game_state = game_state
        self.audience_memory = {}  # username -> interactions
        self.question_history = {}  # username -> questions
        self.topic_interests = {}  # topic -> interest score
        self.active_users = deque(maxlen=100)  # Recently active users
        self.sentiment_tracker = {}  # username -> sentiment history
        self.question_prioritization = True  # Enable question prioritization
        self.personalization = True  # Enable personalization
        
        # Interaction stats
        self.stats = {
            "total_questions": 0,
            "answered_questions": 0,
            "total_users": 0,
            "returning_users": 0,
            "avg_response_time": 0,
            "response_times": deque(maxlen=50)
        }
    
    def add_user_question(self, user_id: str, username: str, question: str) -> Dict[str, Any]:
        """
        Add a question from a user with enhanced tracking
        
        Args:
            user_id: User ID
            username: Username
            question: Question text
            
        Returns:
            Question data with priority and position
        """
        # Track in active users
        self.active_users.append(username)
        
        # Update total stats
        self.stats["total_questions"] += 1
        
        # Check if new user
        if username not in self.audience_memory:
            self.audience_memory[username] = {
                "first_seen": time.time(),
                "interaction_count": 0,
                "questions": [],
                "topics_asked": Counter(),
                "sentiment": 0,
                "user_id": user_id
            }
            self.stats["total_users"] += 1
        else:
            self.stats["returning_users"] += 1
        
        # Update user data
        self.audience_memory[username]["interaction_count"] += 1
        self.audience_memory[username]["last_interaction"] = time.time()
        
        # Analyze question for topics
        topics = self._extract
