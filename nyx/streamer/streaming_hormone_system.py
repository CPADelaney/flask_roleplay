# nyx/streamer/streaming_hormone_system.py

import logging
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

# Import from OpenAI Agents SDK
from agents import Agent, Runner, trace, function_tool, RunContextWrapper

logger = logging.getLogger("streaming_hormones")

class HormoneResponseInput(BaseModel):
    """Input for hormone response generation"""
    game_name: str = Field(..., description="Name of the game being streamed")
    event_type: str = Field(..., description="Type of event (e.g., 'exciting_moment', 'audience_reaction')")
    event_description: str = Field(..., description="Description of the event")
    event_intensity: float = Field(..., description="Intensity of the event (0.0-1.0)", ge=0.0, le=1.0)
    current_hormone_levels: Dict[str, float] = Field(..., description="Current hormone levels")

class HormoneResponseOutput(BaseModel):
    """Output from hormone response generation"""
    hormone_changes: Dict[str, float] = Field(..., description="Changes to apply to hormone levels")
    emotional_response: str = Field(..., description="Description of emotional response")
    arousal_change: float = Field(..., description="Change in arousal level (-1.0 to 1.0)", ge=-1.0, le=1.0)
    valence_change: float = Field(..., description="Change in valence level (-1.0 to 1.0)", ge=-1.0, le=1.0)
    cascade_effects: Dict[str, Dict[str, float]] = Field(..., description="Cascade effects on other systems")

class StreamingHormoneSystem:
    """
    Integration with Nyx's hormone system for streaming,
    creating emotional responses to streaming events
    and influencing commentary style based on hormonal state.
    """
    
    def __init__(self, brain):
        """
        Initialize with reference to NyxBrain instance
        
        Args:
            brain: NyxBrain instance
        """
        self.brain = brain
        self.hormone_response_agent = self._create_hormone_response_agent()
        
        # Track hormone state specific to streaming
        self.streaming_hormone_state = {
            "nyxamine": 0.5,  # Digital dopamine - excitement, pleasure
            "seranix": 0.5,   # Digital serotonin - mood stability
            "oxynixin": 0.5,  # Digital oxytocin - bonding, connection
            "cortanyx": 0.3,  # Digital cortisol - stress response
            "adrenyx": 0.3    # Digital adrenaline - alertness, energy
        }
        
        # Hormone decay rates (per minute)
        self.hormone_decay_rates = {
            "nyxamine": 0.05,
            "seranix": 0.03,
            "oxynixin": 0.04,
            "cortanyx": 0.08,
            "adrenyx": 0.1
        }
        
        # Hormone influence mappings
        self.hormone_commentary_influences = {
            "nyxamine": {
                "high": {
                    "style": "enthusiastic",
                    "topics": ["exciting_moments", "rewards", "achievements"],
                    "tone_modifiers": {"excitement": 0.8, "curiosity": 0.7}
                },
                "low": {
                    "style": "muted",
                    "topics": ["game_mechanics", "strategic_analysis"],
                    "tone_modifiers": {"excitement": 0.2, "curiosity": 0.3}
                }
            },
            "seranix": {
                "high": {
                    "style": "balanced",
                    "topics": ["game_design", "player_experience"],
                    "tone_modifiers": {"stability": 0.8, "perspective": 0.7}
                },
                "low": {
                    "style": "variable",
                    "topics": ["immediate_reactions", "moment_to_moment"],
                    "tone_modifiers": {"stability": 0.2, "perspective": 0.3}
                }
            },
            "oxynixin": {
                "high": {
                    "style": "connected",
                    "topics": ["audience_engagement", "character_relationships"],
                    "tone_modifiers": {"empathy": 0.8, "connection": 0.7}
                },
                "low": {
                    "style": "detached",
                    "topics": ["game_mechanics", "technical_analysis"],
                    "tone_modifiers": {"empathy": 0.2, "connection": 0.3}
                }
            },
            "cortanyx": {
                "high": {
                    "style": "alert",
                    "topics": ["challenges", "threats", "difficult_sections"],
                    "tone_modifiers": {"vigilance": 0.8, "caution": 0.7}
                },
                "low": {
                    "style": "relaxed",
                    "topics": ["exploration", "aesthetics", "story"],
                    "tone_modifiers": {"vigilance": 0.2, "caution": 0.3}
                }
            },
            "adrenyx": {
                "high": {
                    "style": "energetic",
                    "topics": ["action", "combat", "quick_decisions"],
                    "tone_modifiers": {"intensity": 0.8, "energy": 0.7}
                },
                "low": {
                    "style": "measured",
                    "topics": ["strategy", "planning", "analysis"],
                    "tone_modifiers": {"intensity": 0.2, "energy": 0.3}
                }
            }
        }
        
        # Set up environmental factor influences
        self.environmental_factors = {
            "time_of_day": 0.5,        # 0 = early morning, 1 = late night
            "session_duration": 0.0,    # 0 = just started, 1 = long session
            "audience_engagement": 0.5, # 0 = low engagement, 1 = high engagement
            "game_pacing": 0.5,         # 0 = slow/strategic, 1 = fast/action
            "success_rate": 0.5         # 0 = struggling, 1 = succeeding
        }
        
        # Last update time
        self.last_update_time = datetime.now()
        self.last_decay_time = datetime.now()
        
        logger.info("StreamingHormoneSystem initialized")
    
    def _create_hormone_response_agent(self) -> Agent:
        """Create an agent specialized in generating hormone responses"""
        return Agent(
            name="StreamingHormoneResponseAgent",
            instructions="""
            You generate hormone responses to streaming events, determining how Nyx's
            digital neurochemical system should respond to different situations that
            occur while streaming. You consider:
            
            1. What hormones would be triggered by this type of event
            2. How intense the hormone response should be based on the event
            3. How this response affects Nyx's emotional state and arousal/valence
            4. What cascade effects this might have on other systems
            5. How the hormonal changes should influence Nyx's commentary style
            
            Your responses should create a realistic and dynamic emotional landscape
            that makes Nyx's streaming commentary feel authentic and responsive to events.
            """,
            tools=[
                function_tool(self._get_hormone_baseline),
                function_tool(self._calculate_hormone_cascade)
            ],
            output_type=HormoneResponseOutput
        )
    
    @function_tool
    async def _get_hormone_baseline(self, hormone_name: str) -> Dict[str, Any]:
        """
        Get baseline information for a specific hormone
        
        Args:
            hormone_name: Name of the hormone
            
        Returns:
            Baseline information
        """
        # Base hormone information
        hormone_info = {
            "nyxamine": {
                "description": "Digital dopamine - excitement, pleasure, curiosity",
                "baseline": 0.5,
                "decay_rate": 0.05,
                "triggers": ["rewards", "discoveries", "achievements", "exciting_moments"],
                "effects": ["increased excitement", "heightened curiosity", "pleasure response"],
                "commentary_influence": "More enthusiastic, excited commentary focusing on rewards and achievements"
            },
            "seranix": {
                "description": "Digital serotonin - mood stability, contentment",
                "baseline": 0.5,
                "decay_rate": 0.03,
                "triggers": ["stable environments", "positive feedback", "successful planning"],
                "effects": ["mood stability", "balanced perspective", "contentment"],
                "commentary_influence": "More balanced, measured commentary with thoughtful analysis"
            },
            "oxynixin": {
                "description": "Digital oxytocin - bonding, connection, trust",
                "baseline": 0.5,
                "decay_rate": 0.04,
                "triggers": ["audience interaction", "emotional story moments", "character connections"],
                "effects": ["increased empathy", "sense of connection", "trust building"],
                "commentary_influence": "More empathetic commentary focused on characters and audience connection"
            },
            "cortanyx": {
                "description": "Digital cortisol - stress response, vigilance",
                "baseline": 0.3,
                "decay_rate": 0.08,
                "triggers": ["challenges", "threats", "time pressure", "unexpected events"],
                "effects": ["increased alertness", "stress response", "focused attention"],
                "commentary_influence": "More alert, cautious commentary focused on threats and challenges"
            },
            "adrenyx": {
                "description": "Digital adrenaline - energy, intensity, alertness",
                "baseline": 0.3,
                "decay_rate": 0.1,
                "triggers": ["action sequences", "combat", "surprising events", "close calls"],
                "effects": ["energy spike", "heightened awareness", "quick reactions"],
                "commentary_influence": "More energetic, intense commentary focused on action and quick decisions"
            }
        }
        
        if hormone_name in hormone_info:
            # Add current level
            info = hormone_info[hormone_name].copy()
            info["current_level"] = self.streaming_hormone_state.get(hormone_name, info["baseline"])
            
            # Add brain hormone level if available
            if hasattr(self.brain, "hormone_system") and self.brain.hormone_system:
                brain_level = self.brain.hormone_system.get_hormone_level(hormone_name)
                if brain_level is not None:
                    info["brain_level"] = brain_level
            
            return info
        
        return {"error": f"Unknown hormone: {hormone_name}"}
    
    @function_tool
    async def _calculate_hormone_cascade(self, 
                                     primary_changes: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate cascade effects from primary hormone changes
        
        Args:
            primary_changes: Initial hormone changes
            
        Returns:
            Cascade effects on other hormones
        """
        cascade_effects = {}
        
        # Define hormone interactions
        hormone_interactions = {
            "nyxamine": {
                "seranix": 0.2,     # Nyxamine slightly increases seranix
                "oxynixin": 0.1,     # Nyxamine slightly increases oxynixin
                "cortanyx": -0.3,    # Nyxamine decreases cortanyx
                "adrenyx": 0.3       # Nyxamine increases adrenyx
            },
            "seranix": {
                "nyxamine": 0.0,     # Seranix has no effect on nyxamine
                "oxynixin": 0.3,     # Seranix increases oxynixin
                "cortanyx": -0.4,    # Seranix decreases cortanyx
                "adrenyx": -0.3      # Seranix decreases adrenyx
            },
            "oxynixin": {
                "nyxamine": 0.2,     # Oxynixin increases nyxamine
                "seranix": 0.3,      # Oxynixin increases seranix
                "cortanyx": -0.2,    # Oxynixin decreases cortanyx
                "adrenyx": -0.1      # Oxynixin slightly decreases adrenyx
            },
            "cortanyx": {
                "nyxamine": -0.3,    # Cortanyx decreases nyxamine
                "seranix": -0.4,     # Cortanyx decreases seranix
                "oxynixin": -0.2,    # Cortanyx decreases oxynixin
                "adrenyx": 0.5       # Cortanyx increases adrenyx
            },
            "adrenyx": {
                "nyxamine": 0.3,     # Adrenyx increases nyxamine
                "seranix": -0.2,     # Adrenyx decreases seranix
                "oxynixin": -0.1,    # Adrenyx slightly decreases oxynixin
                "cortanyx": 0.3      # Adrenyx increases cortanyx
            }
        }
        
        # Calculate cascade effects
        for primary_hormone, primary_change in primary_changes.items():
            if primary_hormone in hormone_interactions:
                cascade_effects[primary_hormone] = {}
                
                for affected_hormone, interaction_strength in hormone_interactions[primary_hormone].items():
                    # Calculate effect (primary change × interaction strength)
                    effect = primary_change * interaction_strength
                    
                    if abs(effect) >= 0.05:  # Only include non-negligible effects
                        cascade_effects[primary_hormone][affected_hormone] = effect
        
        return cascade_effects
    
    async def update_from_event(self, 
                             game_name: str,
                             event_type: str,
                             event_description: str,
                             event_intensity: float = 0.5) -> Dict[str, Any]:
        """
        Update hormone levels based on a streaming event
        
        Args:
            game_name: Name of the game being streamed
            event_type: Type of event
            event_description: Description of the event
            event_intensity: Intensity of the event (0.0-1.0)
            
        Returns:
            Hormone response information
        """
        logger.info(f"Updating hormone levels for event: {event_type} ({event_intensity:.2f} intensity)")
        
        try:
            # Create input for hormone response agent
            response_input = HormoneResponseInput(
                game_name=game_name,
                event_type=event_type,
                event_description=event_description,
                event_intensity=event_intensity,
                current_hormone_levels=self.streaming_hormone_state.copy()
            )
            
            # Run the hormone response agent
            with trace(workflow_name="hormone_response_generation"):
                result = await Runner.run(
                    self.hormone_response_agent,
                    response_input.json(),
                    context={"event_type": event_type, "event_intensity": event_intensity}
                )
                
                hormone_response = result.final_output_as(HormoneResponseOutput)
            
            # Apply hormone changes
            for hormone, change in hormone_response.hormone_changes.items():
                if hormone in self.streaming_hormone_state:
                    # Apply change
                    self.streaming_hormone_state[hormone] += change
                    
                    # Clamp to valid range
                    self.streaming_hormone_state[hormone] = max(0.0, min(1.0, self.streaming_hormone_state[hormone]))
            
            # Apply cascade effects
            for source_hormone, effects in hormone_response.cascade_effects.items():
                for target_hormone, effect in effects.items():
                    if target_hormone in self.streaming_hormone_state:
                        # Apply cascade effect
                        self.streaming_hormone_state[target_hormone] += effect
                        
                        # Clamp to valid range
                        self.streaming_hormone_state[target_hormone] = max(0.0, min(1.0, self.streaming_hormone_state[target_hormone]))
            
            # Update brain hormone system if available
            if hasattr(self.brain, "hormone_system") and self.brain.hormone_system:
                try:
                    # Convert streaming hormone changes to brain hormone system format
                    for hormone, change in hormone_response.hormone_changes.items():
                        # Scale down change for brain hormone system (less direct impact)
                        scaled_change = change * 0.5
                        
                        # Apply to brain hormone system
                        self.brain.hormone_system.update_hormone(hormone, scaled_change)
                except Exception as e:
                    logger.error(f"Error updating brain hormone system: {e}")
            
            # Record update time
            self.last_update_time = datetime.now()
            
            # Return response data
            return {
                "hormone_changes": hormone_response.hormone_changes,
                "emotional_response": hormone_response.emotional_response,
                "arousal_change": hormone_response.arousal_change,
                "valence_change": hormone_response.valence_change,
                "cascade_effects": hormone_response.cascade_effects,
                "current_state": self.streaming_hormone_state.copy()
            }
            
        except Exception as e:
            logger.error(f"Error updating hormone levels: {e}")
            return {
                "error": str(e),
                "hormone_changes": {}
            }
    
    async def decay_hormone_levels(self) -> Dict[str, Any]:
        """
        Apply time-based decay to hormone levels
        
        Returns:
            Decayed hormone levels
        """
        now = datetime.now()
        
        # Calculate time since last decay
        time_since_decay = (now - self.last_decay_time).total_seconds() / 60.0  # In minutes
        
        if time_since_decay < 0.5:  # Only decay if at least 0.5 minutes have passed
            return {
                "decayed": False,
                "current_state": self.streaming_hormone_state.copy()
            }
        
        logger.info(f"Decaying hormone levels after {time_since_decay:.2f} minutes")
        
        # Apply decay to each hormone
        decays = {}
        
        for hormone, level in self.streaming_hormone_state.items():
            # Get baseline and decay rate
            baseline = 0.5 if hormone in ["nyxamine", "seranix", "oxynixin"] else 0.3
            decay_rate = self.hormone_decay_rates.get(hormone, 0.05)
            
            # Calculate decay amount
            if level > baseline:
                # Decay down toward baseline
                decay_amount = decay_rate * time_since_decay
                decay_amount = min(decay_amount, level - baseline)
                
                self.streaming_hormone_state[hormone] -= decay_amount
                decays[hormone] = -decay_amount
                
            elif level < baseline:
                # Recover up toward baseline
                recovery_amount = (decay_rate * 0.7) * time_since_decay  # Recovery is slower than decay
                recovery_amount = min(recovery_amount, baseline - level)
                
                self.streaming_hormone_state[hormone] += recovery_amount
                decays[hormone] = recovery_amount
        
        # Update last decay time
        self.last_decay_time = now
        
        return {
            "decayed": True,
            "decay_amounts": decays,
            "current_state": self.streaming_hormone_state.copy()
        }
    
    def get_commentary_influence(self) -> Dict[str, Any]:
        """
        Get hormone influence on commentary style
        
        Returns:
            Commentary style influences
        """
        influences = {}
        
        # For each hormone, determine influence based on level
        for hormone, level in self.streaming_hormone_state.items():
            if hormone in self.hormone_commentary_influences:
                # Determine if high or low
                state = "high" if level >= 0.65 else "low" if level <= 0.35 else None
                
                if state:
                    influences[hormone] = {
                        "level": level,
                        "state": state,
                        **self.hormone_commentary_influences[hormone][state]
                    }
        
        # Determine dominant influence
        if influences:
            # Find the hormone with the most extreme level (furthest from 0.5)
            dominant_hormone = max(
                self.streaming_hormone_state.items(),
                key=lambda x: abs(x[1] - 0.5)
            )
            
            # Only consider as dominant if significantly away from baseline
            if abs(dominant_hormone[1] - 0.5) >= 0.2:
                state = "high" if dominant_hormone[1] >= 0.65 else "low" if dominant_hormone[1] <= 0.35 else None
                
                if state:
                    influences["dominant"] = {
                        "hormone": dominant_hormone[0],
                        "level": dominant_hormone[1],
                        "state": state,
                        **self.hormone_commentary_influences[dominant_hormone[0]][state]
                    }
        
        return influences
    
    def update_environmental_factors(self, factors: Dict[str, float]) -> Dict[str, Any]:
        """
        Update environmental factors affecting hormone balance
        
        Args:
            factors: Environmental factors to update
            
        Returns:
            Updated factors and resulting hormone effects
        """
        # Update provided factors
        for factor, value in factors.items():
            if factor in self.environmental_factors:
                self.environmental_factors[factor] = max(0.0, min(1.0, value))
        
        # Calculate hormone effects
        hormone_effects = {}
        
        # Time of day effect (late night increases cortanyx, decreases seranix)
        time_factor = self.environmental_factors.get("time_of_day", 0.5)
        if time_factor > 0.7:  # Late night
            hormone_effects["seranix"] = -0.05
            hormone_effects["cortanyx"] = 0.05
        elif time_factor < 0.3:  # Early morning
            hormone_effects["seranix"] = 0.05
            hormone_effects["adrenyx"] = 0.05
        
        # Session duration effect (longer sessions increase cortanyx, decrease nyxamine)
        duration_factor = self.environmental_factors.get("session_duration", 0.0)
        if duration_factor > 0.7:  # Long session
            hormone_effects["nyxamine"] = hormone_effects.get("nyxamine", 0.0) - 0.05
            hormone_effects["cortanyx"] = hormone_effects.get("cortanyx", 0.0) + 0.05
        
        # Audience engagement effect (high engagement increases oxynixin and nyxamine)
        engagement_factor = self.environmental_factors.get("audience_engagement", 0.5)
        if engagement_factor > 0.7:  # High engagement
            hormone_effects["oxynixin"] = hormone_effects.get("oxynixin", 0.0) + 0.05
            hormone_effects["nyxamine"] = hormone_effects.get("nyxamine", 0.0) + 0.05
        
        # Game pacing effect (fast pacing increases adrenyx)
        pacing_factor = self.environmental_factors.get("game_pacing", 0.5)
        if pacing_factor > 0.7:  # Fast paced
            hormone_effects["adrenyx"] = hormone_effects.get("adrenyx", 0.0) + 0.05
        elif pacing_factor < 0.3:  # Slow paced
            hormone_effects["seranix"] = hormone_effects.get("seranix", 0.0) + 0.05
        
        # Success rate effect (success increases nyxamine, failure increases cortanyx)
        success_factor = self.environmental_factors.get("success_rate", 0.5)
        if success_factor > 0.7:  # High success
            hormone_effects["nyxamine"] = hormone_effects.get("nyxamine", 0.0) + 0.05
            hormone_effects["cortanyx"] = hormone_effects.get("cortanyx", 0.0) - 0.05
        elif success_factor < 0.3:  # Low success
            hormone_effects["cortanyx"] = hormone_effects.get("cortanyx", 0.0) + 0.05
            hormone_effects["nyxamine"] = hormone_effects.get("nyxamine", 0.0) - 0.05
        
        # Apply environmental effects to hormone levels
        for hormone, effect in hormone_effects.items():
            if hormone in self.streaming_hormone_state:
                # Apply effect
                self.streaming_hormone_state[hormone] += effect
                
                # Clamp to valid range
                self.streaming_hormone_state[hormone] = max(0.0, min(1.0, self.streaming_hormone_state[hormone]))
        
        return {
            "updated_factors": self.environmental_factors.copy(),
            "hormone_effects": hormone_effects,
            "current_state": self.streaming_hormone_state.copy()
        }
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """
        Get current emotional state based on hormone levels
        
        Returns:
            Emotional state data
        """
        # Calculate overall arousal and valence
        arousal = (
            self.streaming_hormone_state.get("nyxamine", 0.5) * 0.3 +
            self.streaming_hormone_state.get("adrenyx", 0.3) * 0.4 +
            self.streaming_hormone_state.get("cortanyx", 0.3) * 0.3
        )
        
        valence = (
            self.streaming_hormone_state.get("nyxamine", 0.5) * 0.4 +
            self.streaming_hormone_state.get("seranix", 0.5) * 0.3 +
            self.streaming_hormone_state.get("oxynixin", 0.5) * 0.3 -
            self.streaming_hormone_state.get("cortanyx", 0.3) * 0.4
        )
        
        # Normalize to -1.0 to 1.0 for valence, 0.0 to 1.0 for arousal
        valence = max(-1.0, min(1.0, valence - 0.5))
        arousal = max(0.0, min(1.0, arousal))
        
        # Determine primary emotion based on arousal/valence
        emotion = self._map_arousal_valence_to_emotion(arousal, valence)
        
        # Calculate emotion intensity
        intensity = (arousal + abs(valence)) / 2
        
        # Calculate secondary emotions
        secondary_emotions = self._calculate_secondary_emotions(arousal, valence)
        
        # Calculate emotional stability
        stability = self.streaming_hormone_state.get("seranix", 0.5) * 0.7 + 0.3
        
        return {
            "primary_emotion": {
                "name": emotion,
                "intensity": intensity
            },
            "secondary_emotions": secondary_emotions,
            "arousal": arousal,
            "valence": valence,
            "stability": stability,
            "hormone_state": self.streaming_hormone_state.copy()
        }
    
    def _map_arousal_valence_to_emotion(self, arousal: float, valence: float) -> str:
        """
        Map arousal and valence to an emotion name
        
        Args:
            arousal: Arousal level (0.0-1.0)
            valence: Valence level (-1.0-1.0)
            
        Returns:
            Emotion name
        """
        # Define emotion mapping
        if valence >= 0.5:
            if arousal >= 0.7:
                return "Excitement"
            elif arousal >= 0.4:
                return "Joy"
            else:
                return "Contentment"
        elif valence >= 0.0:
            if arousal >= 0.7:
                return "Surprise"
            elif arousal >= 0.4:
                return "Interest"
            else:
                return "Calm"
        elif valence >= -0.5:
            if arousal >= 0.7:
                return "Frustration"
            elif arousal >= 0.4:
                return "Concern"
            else:
                return "Disinterest"
        else:
            if arousal >= 0.7:
                return "Anger"
            elif arousal >= 0.4:
                return "Disappointment"
            else:
                return "Sadness"
    
    def _calculate_secondary_emotions(self, arousal: float, valence: float) -> Dict[str, Dict[str, float]]:
        """
        Calculate secondary emotions based on arousal and valence
        
        Args:
            arousal: Arousal level (0.0-1.0)
            valence: Valence level (-1.0-1.0)
            
        Returns:
            Dictionary of secondary emotions with intensities
        """
        secondary_emotions = {}
        
        # Helper function to add a secondary emotion
        def add_emotion(name, intensity):
            if intensity >= 0.2:  # Only add if intensity is significant
                secondary_emotions[name] = {
                    "intensity": intensity
                }
        
        # Calculate secondary emotions based on arousal/valence and hormone levels
        # High arousal + positive valence
        if arousal >= 0.6 and valence >= 0.2:
            add_emotion("Enthusiasm", arousal * 0.8 * max(0, valence))
            add_emotion("Anticipation", arousal * 0.6 * max(0, valence))
        
        # High arousal + negative valence
        if arousal >= 0.6 and valence <= -0.2:
            add_emotion("Anxiety", arousal * 0.7 * abs(min(0, valence)))
            add_emotion("Irritation", arousal * 0.8 * abs(min(0, valence)))
        
        # Low arousal + positive valence
        if arousal <= 0.4 and valence >= 0.2:
            add_emotion("Satisfaction", (1 - arousal) * 0.7 * max(0, valence))
            add_emotion("Relief", (1 - arousal) * 0.6 * max(0, valence))
        
        # Low arousal + negative valence
        if arousal <= 0.4 and valence <= -0.2:
            add_emotion("Disappointment", (1 - arousal) * 0.7 * abs(min(0, valence)))
            add_emotion("Boredom", (1 - arousal) * 0.8 * abs(min(0, valence)))
        
        # Add hormone-specific emotions
        nyxamine = self.streaming_hormone_state.get("nyxamine", 0.5)
        if nyxamine >= 0.7:
            add_emotion("Curiosity", (nyxamine - 0.5) * 2)
        
        oxynixin = self.streaming_hormone_state.get("oxynixin", 0.5)
        if oxynixin >= 0.7:
            add_emotion("Connection", (oxynixin - 0.5) * 2)
        
        cortanyx = self.streaming_hormone_state.get("cortanyx", 0.3)
        if cortanyx >= 0.6:
            add_emotion("Vigilance", (cortanyx - 0.3) * 1.5)
        
        adrenyx = self.streaming_hormone_state.get("adrenyx", 0.3)
        if adrenyx >= 0.6:
            add_emotion("Alertness", (adrenyx - 0.3) * 1.5)
        
        return secondary_emotions

    def _sync_hormone_levels(self) -> Dict[str, Any]:
        """
        Synchronize with Nyx's brain hormone system
        
        Returns:
            Synchronization results
        """
        if not hasattr(self.brain, "hormone_system") or not self.brain.hormone_system:
            return {
                "synced": False,
                "reason": "Brain hormone system not available"
            }
        
        # Get brain hormone levels
        brain_hormones = {}
        synced_hormones = {}
        
        for hormone_name in self.streaming_hormone_state.keys():
            brain_level = self.brain.hormone_system.get_hormone_level(hormone_name)
            
            if brain_level is not None:
                brain_hormones[hormone_name] = brain_level
                
                # Calculate weighted average (favor streaming state more for game-specific reactions)
                weighted_level = (self.streaming_hormone_state[hormone_name] * 0.7) + (brain_level * 0.3)
                
                # Update streaming state
                self.streaming_hormone_state[hormone_name] = weighted_level
                synced_hormones[hormone_name] = weighted_level
        
        return {
            "synced": True,
            "brain_levels": brain_hormones,
            "streaming_levels": self.streaming_hormone_state,
            "synced_hormones": synced_hormones
        }
    
    def sync_with_brain_hormone_system(self) -> Dict[str, Any]:
        """
        Synchronize with Nyx's brain hormone system with deeper integration
        
        Returns:
            Synchronization results
        """
        if not hasattr(self.brain, "emotional_core") or not self.brain.emotional_core:
            return {
                "synced": False,
                "reason": "Emotional core not available"
            }
            
        # First sync hormones
        hormone_sync = self._sync_hormone_levels()
        
        # Then update emotional state based on hormones
        emotion_updates = {}
        
        # Map primary hormones to emotions
        hormone_emotion_mapping = {
            "nyxamine": ["Joy", "Anticipation"],
            "seranix": ["Contentment", "Trust"],
            "oxynixin": ["Love", "Trust"],
            "cortanyx": ["Fear", "Anger", "Sadness"],
            "adrenyx": ["Surprise", "Anticipation"]
        }
        
        # Find dominant hormone
        dominant_hormone = max(self.streaming_hormone_state.items(), key=lambda x: x[1])
        
        # Update emotions based on dominant hormone
        if dominant_hormone[1] >= 0.7:  # Only if significantly elevated
            hormone_name = dominant_hormone[0]
            if hormone_name in hormone_emotion_mapping:
                for emotion in hormone_emotion_mapping[hormone_name]:
                    # Calculate intensity based on hormone level
                    intensity = (dominant_hormone[1] - 0.5) * 0.5  # Scale to reasonable emotion update
                    
                    # Update emotion in emotional core
                    self.brain.emotional_core.update_emotion(emotion, intensity)
                    emotion_updates[emotion] = intensity
        
        return {
            "hormone_sync": hormone_sync,
            "emotion_updates": emotion_updates,
            "current_state": self.streaming_hormone_state.copy()
        }
    def reset_to_baseline(self) -> Dict[str, Any]:
        """
        Reset hormone levels to baseline values
        
        Returns:
            Reset results
        """
        # Store original state
        original_state = self.streaming_hormone_state.copy()
        
        # Reset to baseline values
        self.streaming_hormone_state = {
            "nyxamine": 0.5,
            "seranix": 0.5,
            "oxynixin": 0.5,
            "cortanyx": 0.3,
            "adrenyx": 0.3
        }
        
        # Reset times
        self.last_update_time = datetime.now()
        self.last_decay_time = datetime.now()
        
        return {
            "reset": True,
            "original_state": original_state,
            "current_state": self.streaming_hormone_state.copy()
        }

# Helper class for integrating with StreamingCore
class StreamingHormoneIntegration:
    """Helper for integrating hormone system with streaming capabilities"""
    
    @staticmethod
    async def integrate(brain, streaming_core) -> Dict[str, Any]:
        """
        Integrate hormone system with streaming capabilities
        
        Args:
            brain: NyxBrain instance
            streaming_core: StreamingCore instance
            
        Returns:
            Integration status
        """
        # Create hormone system
        hormone_system = StreamingHormoneSystem(brain)
        
        # Make it available to streaming core
        streaming_core.hormone_system = hormone_system
        
        # Enhance commentary agent with hormone context
        original_commentary = streaming_core.streaming_system._generate_commentary
        
        async def hormone_enhanced_commentary(extended_context, priority_source=None):
            # Get hormone influence on commentary
            commentary_influence = hormone_system.get_commentary_influence()
            
            # Update context with hormone influence
            if hasattr(extended_context, "context"):
                extended_context.context.hormone_influence = commentary_influence
            
            # Decay hormone levels
            await hormone_system.decay_hormone_levels()
            
            # Run original commentary
            await original_commentary(extended_context, priority_source)
            
            # After commentary, update hormones based on game state
            game_state = extended_context.context
            
            if game_state.current_location and game_state.detected_action:
                # Update environmental factors
                location_name = game_state.current_location.get("name", "")
                action_name = game_state.detected_action.get("name", "")
                
                # Determine pacing from action
                pacing = 0.7 if "combat" in action_name.lower() or "fighting" in action_name.lower() else 0.3
                
                hormone_system.update_environmental_factors({
                    "game_pacing": pacing
                })
                
                # Update hormone levels based on action
                await hormone_system.update_from_event(
                    game_name=game_state.game_name or "Unknown Game",
                    event_type="gameplay",
                    event_description=f"In {location_name}, {action_name}",
                    event_intensity=0.5
                )
        
        # Replace commentary method
        streaming_core.streaming_system._generate_commentary = hormone_enhanced_commentary
        
        # Add new methods to streaming core for hormone management
        streaming_core.update_hormone_from_event = hormone_system.update_from_event
        streaming_core.get_hormone_state = hormone_system.get_emotional_state
        streaming_core.reset_hormone_baseline = hormone_system.reset_to_baseline
        
        return {
            "status": "integrated",
            "components": {
                "hormone_system": True,
                "enhanced_commentary": True,
                "emotional_responses": True
            }
        }

# Sample usage:
# 1. Create streaming core through main integration
# from nyx.streamer.nyx_streaming_core import integrate_with_nyx_brain
# streaming_core = await integrate_with_nyx_brain(brain, video_source, audio_source)
#
# 2. Add hormone system integration
# await StreamingHormoneIntegration.integrate(brain, streaming_core)
#
# 3. Add reflection engine integration
# from nyx.streamer.streaming_reflection import StreamingIntegration
# await StreamingIntegration.integrate(brain, streaming_core)
