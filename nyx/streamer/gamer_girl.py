# /nyx/streamer/gamer_girl.py

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

# Import game vision module components
from enhanced_game_vision import (
    GameKnowledgeBase, EnhancedGameRecognitionSystem,
    EnhancedSpatialMemory, SceneGraphAnalyzer,
    GameActionRecognition, RealTimeGameProcessor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("game_agents")

# Set OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY environment variable must be set")

class EnhancedMultiModalIntegrator:
    """
    Enhanced integration between visual, audio, and speech modalities
    for more comprehensive game understanding and commentary.
    """
    
    def __init__(self, game_state: GameState):
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
                        print(f"Error in frame comparison: {e}")
        
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
    # This function would typically use a web search tool from the Agents SDK
    # For this implementation, we're using a placeholder
    
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

###########################################
# Enhanced Commentary Agent
###########################################

class CommentaryType(BaseModel):
    commentary: str = Field(..., description="The gameplay commentary to share with viewers")
    focus: Literal["gameplay", "strategy", "lore", "mechanics", "dialog", "cross_game"] = Field(
        ..., description="What aspect of the game the commentary focuses on"
    )
    source: Literal["visual", "audio", "speech", "cross_game"] = Field(
        ..., description="The primary source of information for this commentary"
    )

enhanced_commentary_agent = Agent(
    name="Enhanced Gameplay Commentator",
    instructions="""
    You are an expert gameplay commentator who provides insightful, entertaining commentary 
    about the game being played. Your commentary should be engaging and relevant to what's 
    happening in the game right now, drawing on both visual and audio information.
    
    Important guidelines:
    - Keep commentary concise and to the point (1-2 sentences)
    - Sound natural and conversational, like a Twitch streamer
    - Vary between different types of focus (gameplay, strategy, lore, mechanics, dialog, cross-game)
    - Consider multiple input sources:
      - Visual elements (objects, locations, actions)
      - Audio cues (sound effects, music)
      - Speech and dialog (character conversations)
      - Cross-game knowledge (insights from similar games)
    - Pay special attention to dialog and storytelling moments
    - Compare mechanics and features to similar games when relevant
    - Don't be repetitive - vary your commentary style
    - Express excitement when interesting events happen
    - Use appropriate gaming terminology
    - Apply learning tips provided by the system to improve your commentary
    
    Remember: You're watching this in real-time with the audience, so make it engaging!
    """,
    output_type=CommentaryType,
    tools=[
        analyze_current_frame,
        analyze_speech,
        get_recent_dialog,
        find_similar_games,
        get_cross_game_insights,
        search_game_information
    ],
    model_settings=ModelSettings(
        temperature=0.7  # More creative commentary
    )
)

###########################################
# Enhanced Question Answering Agent
###########################################

class AnswerType(BaseModel):
    answer: str = Field(..., description="The answer to the audience question")
    relevance: int = Field(
        ..., 
        description="How relevant this question is to the current gameplay (1-10 scale)",
        ge=1, 
        le=10
    )
    used_sources: List[Literal["visual", "audio", "speech", "cross_game", "web_search"]] = Field(
        ..., description="The sources of information used to answer this question"
    )

enhanced_question_agent = Agent(
    name="Enhanced Question Answerer",
    instructions="""
    You answer questions from the audience about the game being played. Your answers should be:
    
    - Concise and focused on the question asked
    - Informative, drawing on multiple sources of information:
      - Visual elements in the game
      - Audio and speech content
      - Knowledge of similar games
      - Web search results when needed
    - Enhanced with appropriate comparisons to similar games
    - Friendly and conversational in tone
    - Improved based on learning tips from past feedback
    
    For each question, also rate how relevant it is to what's currently happening in the game
    on a scale of 1-10 (with 10 being extremely relevant). This helps the system decide when
    to show your answer.
    
    Some questions may be off-topic or inappropriate. For these, provide a brief, polite response
    indicating you're focusing on game-related questions.
    """,
    output_type=AnswerType,
    tools=[
        analyze_current_frame,
        analyze_speech,
        get_recent_dialog,
        find_similar_games,
        get_cross_game_insights,
        search_game_information
    ],
    model_settings=ModelSettings(
        temperature=0.4  # More factual answers
    )
)

###########################################
# Enhanced Triage Agent for Orchestration
###########################################

class TriageDecision(BaseModel):
    choice: Literal["commentary", "answer_question", "skip"] = Field(
        ..., 
        description="Which action to take: provide commentary, answer a question, or skip this frame"
    )
    reasoning: str = Field(..., description="Reasoning behind the decision")
    priority_source: Optional[Literal["visual", "audio", "speech", "cross_game"]] = Field(
        None, description="The primary source that should be prioritized in the response"
    )

enhanced_triage_agent = Agent(
    name="Enhanced Triage Agent",
    instructions="""
    You are an orchestration agent that decides whether to provide commentary or answer 
    audience questions based on the current game state. You should:
    
    - Prioritize commentary when significant game events occur:
      - Important visual changes
      - Notable audio events (music changes, significant sound effects)
      - Character dialog or story moments
      - Cross-game insights that apply to the current situation
    - Answer questions when they are relevant to current gameplay or have been waiting too long
    - Avoid overwhelming the viewer with too much text at once
    - Ensure there's a good balance between different types of commentary
    - Indicate which source of information (visual, audio, speech, cross-game) should be prioritized
    - Prioritize higher-relevance questions over lower-relevance ones
    
    Your job is to create a balanced viewing experience that mixes insightful commentary
    with timely answers to audience questions, while taking advantage of all information sources.
    """,
    output_type=TriageDecision,
    handoffs=[
        handoff(enhanced_commentary_agent, tool_name_override="provide_commentary"),
        handoff(enhanced_question_agent, tool_name_override="answer_question")
    ],
    model_settings=ModelSettings(
        temperature=0.2  # More consistent decision making
    )
)

###########################################
# Advanced Game Agent System
###########################################

class AdvancedGameAgentSystem:
    """
    Advanced system that integrates multi-modal game analysis with
    speech recognition and cross-game knowledge for enhanced streaming experience.
    """
    
    def __init__(self, video_source=0, audio_source=None):
        """Initialize the system with video and audio sources"""
        self.video_source = video_source
        self.audio_source = audio_source
        self.game_state = GameState()
        self.last_commentary_time = time.time()
        self.commentary_cooldown = 5.0  # seconds between commentaries
        
        # Initialize knowledge base
        self.knowledge_base = GameKnowledgeBase()
        
        # Initialize audio processor
        self.audio_processor = GameAudioProcessor()
        
        # Initialize speech recognition
        self.speech_recognition = SpeechRecognitionSystem()
        
        # Initialize cross-game knowledge
        self.cross_game_knowledge = CrossGameKnowledgeSystem()
        
        # Seed initial cross-game knowledge
        self.cross_game_knowledge.seed_initial_knowledge()
        
        # Initialize web search tool
        self.web_search_tool = WebSearchTool()
        
        # Set up game processor
        self.game_processor = RealTimeGameProcessor(
            game_system=None,  # We handle processing ourselves
            input_source=video_source,
            processing_fps=30
        )
        
        self.multi_modal_integrator = EnhancedMultiModalIntegrator(self.game_state)
        
        # Set up flags
        self.running = False
        self.processing_frame = False
        
        # Track processing metrics
        self.frame_times = deque(maxlen=100)
        self.agent_times = deque(maxlen=20)
        
        logger.info("AdvancedGameAgentSystem initialized")
    
    async def start(self):
        """Start the system"""
        logger.info("Starting AdvancedGameAgentSystem")
        self.running = True
        
        # Start audio capture
        self.audio_processor.start_capture(self.audio_source)
        
        # Start speech recognition
        self.speech_recognition.start_capture(self.audio_source)
        
        # Start video processing
        await self._process_video_stream()
    
    async def stop(self):
        """Stop the system"""
        logger.info("Stopping AdvancedGameAgentSystem")
        self.running = False
        
        # Stop audio capture
        self.audio_processor.stop_capture()
        
        # Stop speech recognition
        self.speech_recognition.stop_capture()
        
        # Stop video processing
        if self.game_processor:
            self.game_processor.stop_processing()
        
        # Save cross-game knowledge
        self.cross_game_knowledge.save_knowledge()
    
    async def _process_video_stream(self):
        """Process the video stream in real-time"""
        # Open video capture
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {self.video_source}")
            return
        
        logger.info(f"Successfully opened video source: {self.video_source}")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    await asyncio.sleep(0.1)
                    continue
                
                # Generate some synthetic audio for testing
                # In a real implementation, this would come from the game audio
                audio_data = np.random.randn(1600)  # 0.1s of audio at 16kHz
                
                # Update game state
                self.game_state.current_frame = frame
                self.game_state.current_audio = audio_data
                self.game_state.frame_timestamp = time.time()
                self.game_state.frame_count += 1
                
                # Process audio
                self.audio_processor.process_audio_block()
                
                # Add audio to speech recognition
                self.speech_recognition.add_audio_data(audio_data)
                
                # Only process certain frames to avoid overwhelming the system
                if self.game_state.frame_count % 30 == 0:  # Process every 30th frame
                    await self._process_game_frame()
                
                # Record frame processing time
                frame_time = time.time() - start_time
                self.frame_times.append(frame_time)
                
                # Sleep to maintain target FPS
                sleep_time = max(0, 1/30 - frame_time)
                await asyncio.sleep(sleep_time)
                
        finally:
            cap.release()
            logger.info("Video capture released")
    
    async def _process_game_frame(self):
        """Process a game frame with advanced multi-modal agent analysis"""
        if self.processing_frame:
            return  # Skip if already processing
        
        self.processing_frame = True
        start_time = time.time()
        
        try:
            # Create context for tools with access to all systems
            extended_context = RunContextWrapper(context=self.game_state)
            extended_context.audio_processor = self.audio_processor
            extended_context.speech_recognition = self.speech_recognition
            extended_context.cross_game_knowledge = self.cross_game_knowledge
            extended_context.web_search_tool = self.web_search_tool
            
            # Only identify game if not already identified
            if not self.game_state.game_id:
                logger.info("Identifying game...")
                await identify_game(extended_context)
                
                # If game identified, find similar games
                if self.game_state.game_id:
                    await find_similar_games(extended_context)
            
            # Always update multi-modal data
            if self.game_state.game_id:
                logger.info("Analyzing frame, audio, and speech...")
                
                # Process in parallel
                tasks = [
                    analyze_current_frame(extended_context),
                    analyze_speech(extended_context),
                    get_player_location(extended_context),
                    detect_current_action(extended_context)
                ]
                await asyncio.gather(*tasks)
            
            # Check if it's time for commentary or to answer a question
            current_time = time.time()
            time_since_last = current_time - self.last_commentary_time
            
            if time_since_last >= self.commentary_cooldown:
                # Use triage agent to decide what to do
                logger.info("Running triage agent...")
                with trace("GameStream", workflow_name="advanced_triage_decision"):
                    # Pass extended context to the triage agent
                    triage_result = await Runner.run(
                        enhanced_triage_agent, 
                        "Decide what to do next based on multi-modal analysis and cross-game insights.", 
                        context=self.game_state
                    )
                    decision = triage_result.final_output_as(TriageDecision)
                
                logger.info(f"Triage decision: {decision.choice} (Priority source: {decision.priority_source or 'none'})")
                
                if decision.choice == "commentary":
                    await self._generate_commentary(extended_context, decision.priority_source)
                elif decision.choice == "answer_question" and self.game_state.pending_questions:
                    await self._answer_question(extended_context)
                # Skip otherwise
                
                self.last_commentary_time = current_time
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        finally:
            processing_time = time.time() - start_time
            self.agent_times.append(processing_time)
            self.processing_frame = False
            logger.info(f"Frame processed in {processing_time:.3f}s")
    
    async def _generate_commentary(self, extended_context, priority_source=None):
        """Generate commentary based on multi-modal analysis and cross-game insights"""
        logger.info(f"Generating commentary with priority source: {priority_source or 'none'}")
        
        # Prepare instruction based on priority source
        if priority_source == "visual":
            instruction = "Provide commentary focusing on the visual elements in the current frame."
        elif priority_source == "audio":
            instruction = "Provide commentary focusing on the audio cues and sound effects."
        elif priority_source == "speech":
            instruction = "Provide commentary focusing on the character dialog and speech."
        elif priority_source == "cross_game":
            instruction = "Provide commentary comparing this game to similar games."
        else:
            instruction = "Provide commentary on the current gameplay using all available information."
        
        with trace("GameStream", workflow_name="advanced_commentary"):
            # Run the commentary agent with the extended context
            result = await Runner.run(
                enhanced_commentary_agent, 
                instruction, 
                context=self.game_state
            )
            commentary = result.final_output_as(CommentaryType)
        
        logger.info(f"Commentary generated: {commentary.commentary} (Focus: {commentary.focus}, Source: {commentary.source})")
        # In a real system, this would be sent to the streaming output
        print(f"\n[COMMENTARY ({commentary.focus.upper()}, {commentary.source.upper()})] {commentary.commentary}\n")
    
    async def _answer_question(self, extended_context):
        """Answer the next audience question using multi-modal analysis and cross-game insights"""
        # Get the next question from the queue
        question_data = self.game_state.get_next_question()
        if not question_data:
            logger.info("No questions to answer")
            return
        
        logger.info(f"Answering question from {question_data['username']}")
        question = question_data['question']
        
        try:
            with trace("GameStream", workflow_name="advanced_question_answering"):
                result = await Runner.run(
                    enhanced_question_agent, 
                    question, 
                    context=self.game_state
                )
                answer = result.final_output_as(AnswerType)
            
            logger.info(f"Question answered (relevance: {answer.relevance}/10, sources: {answer.used_sources})")
            
            # Store the answered question in the game state
            self.game_state.add_answered_question(question_data, answer.answer)
            
            # In a real system, this would be sent to the streaming output
            print(f"\n[Q&A] {question_data['username']} asked: {question}")
            print(f"[ANSWER (using {', '.join(answer.used_sources)})] {answer.answer}\n")
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            # In a real system, this might be logged but not shown to the audience
    
    def add_audience_question(self, user_id: str, username: str, question: str):
        """Add a question from the audience"""
        logger.info(f"Question received from {username}: {question}")
        self.game_state.add_question(user_id, username, question)
    
    def get_cross_game_insights(self, target_game: str, context: Optional[str] = None):
        """Get insights from other games for the specified game"""
        return self.cross_game_knowledge.get_applicable_insights(target_game, context)
    
    def get_performance_metrics(self):
        """Get system performance metrics"""
        avg_frame_time = sum(self.frame_times) / max(len(self.frame_times), 1)
        avg_agent_time = sum(self.agent_times) / max(len(self.agent_times), 1)
        
        return {
            "fps": 1.0 / max(avg_frame_time, 0.001),
            "avg_frame_processing_time": avg_frame_time,
            "avg_agent_processing_time": avg_agent_time,
            "frame_count": self.game_state.frame_count,
            "pending_questions": len(self.game_state.pending_questions),
            "speech_transcriptions": len(self.game_state.dialog_history),
            "similar_games": len(self.game_state.similar_games),
            "transferred_insights": len(self.game_state.transferred_insights)
        }

    async def _process_game_frame(self):
        """Process a game frame with advanced multi-modal agent analysis"""
        if self.processing_frame:
            return  # Skip if already processing
        
        self.processing_frame = True
        start_time = time.time()
        
        try:
            # Create context for tools with access to all systems
            extended_context = RunContextWrapper(context=self.game_state)
            extended_context.audio_processor = self.audio_processor
            extended_context.speech_recognition = self.speech_recognition
            extended_context.cross_game_knowledge = self.cross_game_knowledge
            extended_context.web_search_tool = self.web_search_tool
            
            # Process all modalities
            await self._process_modalities(extended_context)
            
            # Integrate multi-modal processing - NEW!
            combined_events = await self.multi_modal_integrator.process_frame(
                self.game_state.current_frame,
                self.game_state.current_audio
            )
            
            # Process any detected multi-modal events
            for event in combined_events:
                if event["data"].get("significance", 0) >= 7.0:
                    # This is a significant event, prioritize for commentary
                    self.game_state.add_event("significant_moment", event["data"])
            
            # Check if it's time for commentary or to answer a question
            current_time = time.time()
            time_since_last = current_time - self.last_commentary_time
            
            if time_since_last >= self.commentary_cooldown:
                # Use triage agent to decide what to do
                with trace("GameStream", workflow_name="advanced_triage_decision"):
                    # Pass extended context to the triage agent
                    triage_result = await Runner.run(
                        enhanced_triage_agent, 
                        "Decide what to do next based on multi-modal analysis and cross-game insights.", 
                        context=self.game_state
                    )
                    decision = triage_result.final_output
                
                if decision.choice == "commentary":
                    await self._generate_commentary(extended_context, decision.priority_source)
                elif decision.choice == "answer_question" and self.game_state.pending_questions:
                    await self._answer_question(extended_context)
                # Skip otherwise
                
                self.last_commentary_time = current_time
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        finally:
            processing_time = time.time() - start_time
            self.agent_times.append(processing_time)
            self.processing_frame = False
            
    async def _process_modalities(self, extended_context):
        """Process all modalities in parallel"""
        # Only identify game if not already identified
        if not self.game_state.game_id:
            await identify_game(extended_context)
            
            # If game identified, find similar games
            if self.game_state.game_id:
                await find_similar_games(extended_context)
        
        # Always update multi-modal data if game is identified
        if self.game_state.game_id:
            # Process in parallel
            tasks = [
                analyze_current_frame(extended_context),
                analyze_speech(extended_context),
                get_player_location(extended_context),
                detect_current_action(extended_context)
            ]
            results = await asyncio.gather(*tasks)
            
            # Process speech results for multi-modal integration
            speech_result = results[1]
            if "Transcribed speech" in speech_result:
                # Extract the transcribed text
                text = speech_result.split('"')[1] if '"' in speech_result else ""
                confidence = float(speech_result.split("Confidence: ")[1].split(")")[0]) if "Confidence: " in speech_result else 0.8
                speaker = speech_result.split("from ")[1].split(":")[0] if "from " in speech_result else None
                
                # Add to multi-modal integrator
                await self.multi_modal_integrator.add_speech_event({
                    "text": text,
                    "confidence": confidence,
                    "speaker": speaker
                })
# Add to nyx/streamer/gamer_girl.py
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
        topics = self._extract_topics(question)
        for topic in topics:
            self.audience_memory[username]["topics_asked"][topic] += 1
            
            # Update global topic interests
            if topic not in self.topic_interests:
                self.topic_interests[topic] = 0
            self.topic_interests[topic] += 1
        
        # Calculate priority
        priority = self._calculate_question_priority(username, question, topics)
        
        # Create question data
        question_data = {
            "user_id": user_id,
            "username": username,
            "question": question,
            "topics": topics,
            "timestamp": time.time(),
            "priority": priority,
            "personalize": self.personalization
        }
        
        # Add to user's question history
        if username not in self.question_history:
            self.question_history[username] = []
        
        self.question_history[username].append(question_data)
        self.audience_memory[username]["questions"].append(question)
        
        # Add to game state queue
        self.game_state.add_question(user_id, username, question)
        
        return question_data
    
    def record_question_answered(self, question_data: Dict[str, Any], answer: str) -> Dict[str, Any]:
        """
        Record that a question was answered
        
        Args:
            question_data: Question data
            answer: Answer text
            
        Returns:
            Updated stats
        """
        # Calculate response time
        if "timestamp" in question_data:
            response_time = time.time() - question_data["timestamp"]
            self.stats["response_times"].append(response_time)
            self.stats["avg_response_time"] = sum(self.stats["response_times"]) / len(self.stats["response_times"])
        
        # Update stats
        self.stats["answered_questions"] += 1
        
        # Store the answer
        username = question_data.get("username")
        if username in self.audience_memory:
            if "answers_received" not in self.audience_memory[username]:
                self.audience_memory[username]["answers_received"] = []
            
            self.audience_memory[username]["answers_received"].append({
                "question": question_data.get("question", ""),
                "answer": answer,
                "timestamp": time.time()
            })
        
        return {
            "stats": self.stats,
            "username": username,
            "user_data": self.audience_memory.get(username)
        }
    
    def get_user_personalization(self, username: str) -> Dict[str, Any]:
        """
        Get personalization data for a user
        
        Args:
            username: Username
            
        Returns:
            Personalization data
        """
        if not self.personalization or username not in self.audience_memory:
            return {}
        
        user_data = self.audience_memory[username]
        
        # Calculate top interests
        top_topics = user_data["topics_asked"].most_common(3)
        
        # Calculate interaction frequency
        interaction_count = user_data["interaction_count"]
        first_seen = user_data.get("first_seen", time.time())
        interaction_period = (time.time() - first_seen) / 86400  # days
        frequency = interaction_count / max(1, interaction_period)
        
        # Determine user type
        user_type = "new"
        if interaction_count >= 10:
            user_type = "regular"
        elif interaction_count >= 3:
            user_type = "returning"
        
        return {
            "username": username,
            "interaction_count": interaction_count,
            "top_interests": top_topics,
            "frequency": frequency,
            "user_type": user_type,
            "questions_asked": len(user_data.get("questions", [])),
            "answers_received": len(user_data.get("answers_received", []))
        }
    
    def _extract_topics(self, question: str) -> List[str]:
        """
        Extract topics from a question
        
        Args:
            question: Question text
            
        Returns:
            List of topics
        """
        # Simple keyword-based topic extraction
        topics = []
        
        # Game mechanics
        if any(word in question.lower() for word in ["mechanics", "controls", "gameplay", "play", "system"]):
            topics.append("game_mechanics")
        
        # Story/plot
        if any(word in question.lower() for word in ["story", "plot", "character", "narrative", "lore"]):
            topics.append("story")
        
        # Strategy/tips
        if any(word in question.lower() for word in ["strategy", "tips", "how to", "best way", "help"]):
            topics.append("strategy")
        
        # Technical
        if any(word in question.lower() for word in ["graphics", "performance", "technical", "lag", "bug"]):
            topics.append("technical")
        
        # Comparisons
        if any(word in question.lower() for word in ["compare", "better", "different", "like", "similar"]):
            topics.append("comparisons")
        
        # If no specific topics found, use "general"
        if not topics:
            topics.append("general")
        
        return topics
    
    def _calculate_question_priority(self, username: str, question: str, topics: List[str]) -> float:
        """
        Calculate question priority
        
        Args:
            username: Username
            question: Question text
            topics: Question topics
            
        Returns:
            Priority score (0-1)
        """
        if not self.question_prioritization:
            return 0.5  # Default priority
        
        base_priority = 0.5
        priority_modifiers = []
        
        # User engagement factor
        if username in self.audience_memory:
            user_data = self.audience_memory[username]
            
            # Reward regulars slightly
            if user_data["interaction_count"] > 5:
                priority_modifiers.append(0.1)
            
            # But also prioritize first-time askers
            if user_data["interaction_count"] == 1:
                priority_modifiers.append(0.2)
            
            # Check waiting time
            if len(user_data["questions"]) > len(user_data.get("answers_received", [])):
                # User has unanswered questions
                priority_modifiers.append(0.15)
        
        # Topic relevance factor
        current_game = self.game_state.game_name
        current_action = self.game_state.detected_action.get("name") if self.game_state.detected_action else None
        
        # Check if question is relevant to current game/action
        if current_game and current_game.lower() in question.lower():
            priority_modifiers.append(0.2)
        
        if current_action and current_action.lower() in question.lower():
            priority_modifiers.append(0.25)
        
        # Topics currently being shown
        relevant_topics = []
        if "game_mechanics" in topics and current_action:
            relevant_topics.append("game_mechanics")
        
        if "story" in topics and any(event["type"] == "character_dialog" for event in self.game_state.recent_events):
            relevant_topics.append("story")
        
        if relevant_topics:
            priority_modifiers.append(0.2)
        
        # Calculate final priority
        final_priority = base_priority + sum(priority_modifiers)
        
        # Clamp to valid range
        return max(0.0, min(1.0, final_priority))
    
    def get_popular_topics(self, limit: int = 5) -> List[Tuple[str, int]]:
        """
        Get most popular topics among audience
        
        Args:
            limit: Maximum number of topics to return
            
        Returns:
            List of (topic, count) tuples
        """
        # Get topics sorted by popularity
        sorted_topics = sorted(self.topic_interests.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_topics[:limit]
    
    def get_audience_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive audience statistics
        
        Returns:
            Audience statistics
        """
        # Calculate active users
        active_count = len(set(self.active_users))
        
        # Calculate returning users
        returning_users = sum(1 for user, data in self.audience_memory.items() 
                             if data["interaction_count"] > 1)
        
        # Calculate metrics
        engagement_rate = self.stats["answered_questions"] / max(1, self.stats["total_questions"])
        avg_questions_per_user = self.stats["total_questions"] / max(1, self.stats["total_users"])
        
        # Get top topics
        top_topics = self.get_popular_topics(5)
        
        return {
            "total_users": self.stats["total_users"],
            "active_users": active_count,
            "returning_users": returning_users,
            "total_questions": self.stats["total_questions"],
            "answered_questions": self.stats["answered_questions"],
            "engagement_rate": engagement_rate,
            "avg_questions_per_user": avg_questions_per_user,
            "avg_response_time": self.stats["avg_response_time"],
            "top_topics": top_topics
        }

# Enhance AdvancedGameAgentSystem
class AdvancedGameAgentSystem:
    # ... (existing code)
    
    def __init__(self, video_source=0, audio_source=None):
        # ... (existing initialization)
        
        # Add enhanced audience interaction
        self.enhanced_audience = EnhancedAudienceInteraction(self.game_state)
        
        # ... (rest of existing code)
    
    def add_audience_question(self, user_id: str, username: str, question: str):
        """Add a question from the audience with enhanced tracking"""
        # Use enhanced audience system
        question_data = self.enhanced_audience.add_user_question(user_id, username, question)
        
        # Original functionality still happens via game_state
        return question_data
    
    async def _answer_question(self, extended_context):
        """Answer the next audience question using multi-modal analysis and cross-game insights"""
        # Get the next question from the queue
        question_data = self.game_state.get_next_question()
        if not question_data:
            logger.info("No questions to answer")
            return
        
        logger.info(f"Answering question from {question_data['username']}")
        question = question_data['question']
        
        try:
            # Get personalization data if available
            personalization = {}
            if hasattr(self, "enhanced_audience"):
                personalization = self.enhanced_audience.get_user_personalization(question_data['username'])
                
                # Add personalization to context
                if personalization and hasattr(extended_context, "context"):
                    extended_context.context.user_personalization = personalization
            
            with trace("GameStream", workflow_name="advanced_question_answering"):
                # Run the enhanced question agent
                result = await Runner.run(
                    enhanced_question_agent, 
                    question, 
                    context=self.game_state
                )
                answer = result.final_output
            
            # Store the answered question in the game state
            self.game_state.add_answered_question(question_data, answer.answer)
            
            # Record the answer in audience system
            if hasattr(self, "enhanced_audience"):
                self.enhanced_audience.record_question_answered(question_data, answer.answer)
            
            # In a real system, this would be sent to the streaming output
            print(f"\n[Q&A] {question_data['username']} asked: {question}")
            print(f"[ANSWER (using {', '.join(answer.used_sources)})] {answer.answer}\n")
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            # In a real system, this might be logged but not shown to the audience
