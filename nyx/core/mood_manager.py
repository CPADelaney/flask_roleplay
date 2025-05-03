# nyx/core/mood_manager.py

import logging
import datetime
import math
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

class MoodState(BaseModel):
    """Represents Nyx's current mood."""
    # Dimensional approach: Valence (Pleasant/Unpleasant), Arousal (Active/Passive), Control (Dominant/Submissive)
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Overall pleasantness (-1=Very Unpleasant, 0=Neutral, 1=Very Pleasant)")
    arousal: float = Field(0.5, ge=0.0, le=1.0, description="Overall energy level (0=Calm/Passive, 0.5=Neutral, 1=Excited/Active)")
    control: float = Field(0.0, ge=-1.0, le=1.0, description="Sense of control/dominance (-1=Submissive, 0=Neutral, 1=Dominant)")
    # Descriptive label (derived from dimensions)
    dominant_mood: str = Field("Neutral", description="Primary mood descriptor")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Overall intensity of the mood")
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    influences: Dict[str, float] = Field(default_factory=dict, description="Factors currently influencing the mood")

class MoodHistory(BaseModel):
    """Records a mood state at a specific time for history tracking."""
    timestamp: datetime.datetime
    valence: float
    arousal: float
    control: float
    dominant_mood: str
    intensity: float
    trigger: Optional[str] = None  # What caused this mood state

class EventImpact(BaseModel):
    """Represents the impact of an event on mood."""
    valence_change: float = Field(0.0, ge=-1.0, le=1.0, description="Change in pleasantness")
    arousal_change: float = Field(0.0, ge=-0.5, le=0.5, description="Change in energy level")
    control_change: float = Field(0.0, ge=-1.0, le=1.0, description="Change in sense of control")
    event_type: str = Field(..., description="Type of event causing the mood change")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Intensity of the event's impact")

class MoodManager:
    """Manages Nyx's mid-term affective state (mood) using Agent SDK."""

    def __init__(self, emotional_core=None, hormone_system=None, needs_system=None, goal_manager=None):
        """Initialize the mood manager with dependencies."""
        # Dependencies
        self.emotional_core = emotional_core
        self.hormone_system = hormone_system
        self.needs_system = needs_system
        self.goal_manager = goal_manager

        # State
        self.current_mood = MoodState()
        self.mood_history: List[MoodHistory] = []
        self.max_history = 100
        self._lock = asyncio.Lock()  # Lock for thread-safe operations

        # Configuration
        self.inertia = 0.90  # How much the previous mood state persists (0-1)
        self.update_interval_seconds = 60  # How often to update mood
        self.last_update_time = datetime.datetime.now()

        # Weights for different influences
        self.influence_weights = {
            "emotion_valence": 0.4,  # Direct impact from emotions
            "emotion_arousal": 0.3,  # Arousal from emotions
            "hormones": 0.15,        # Impact from hormone system
            "needs": 0.1,            # Impact from needs satisfaction
            "goals": 0.05,           # Impact from goal outcomes
            "external_events": 0.2,  # Impact from significant external events
        }

        # Create and set up the agent
        self.agent = self._create_agent()
        
        logger.info("MoodManager initialized")

    def _create_agent(self) -> Agent:
        """Create the agent with the necessary tools."""
        agent = Agent(
            name="Mood Manager",
            instructions="""You manage the AI's emotional state and mood, responsible for:
        1. Updating mood based on various inputs (emotions, hormones, needs, goals)
        2. Tracking mood history
        3. Responding to significant events
        4. Providing insights into current mood state
        
        Use the appropriate tools to perform these tasks and maintain the AI's affective state.
        """,
            tools=[
                self.update_mood,
                self.get_current_mood,
                self.modify_mood,
                self.handle_significant_event,
                self.get_mood_history
            ]
        )
        return agent

    # Helper methods
    def _get_dominant_mood_label(self, v: float, a: float, c: float) -> str:
        """Maps mood dimensions to a descriptive label."""
        # Simple mapping based on quadrants/octants of V-A-C space
        if a > 0.7:  # High Arousal
            if v > 0.3: 
                return "Excited" if c > 0.3 else ("Elated" if c < -0.3 else "Enthusiastic")
            if v < -0.3: 
                return "Anxious" if c > 0.3 else ("Stressed" if c < -0.3 else "Tense")
            return "Alert" if c > 0.3 else ("Agitated" if c < -0.3 else "Focused")
            
        elif a < 0.3:  # Low Arousal
            if v > 0.3: 
                return "Calm" if c > 0.3 else ("Relaxed" if c < -0.3 else "Content")
            if v < -0.3: 
                return "Depressed" if c > 0.3 else ("Sad" if c < -0.3 else "Bored")
            return "Passive" if c > 0.3 else ("Lethargic" if c < -0.3 else "Drowsy")
            
        else:  # Mid Arousal
            if v > 0.5: 
                return "Happy" if c > 0.3 else ("Pleased" if c < -0.3 else "Glad")
            if v < -0.5: 
                return "Unhappy" if c > 0.3 else ("Displeased" if c < -0.3 else "Gloomy")
            return "Neutral" if abs(c) < 0.3 else ("Confident" if c > 0 else "Reserved")
    
    def _add_to_history(self, trigger: Optional[str] = None) -> None:
        """Add current mood to history."""
        history_entry = MoodHistory(
            timestamp=self.current_mood.last_updated,
            valence=self.current_mood.valence,
            arousal=self.current_mood.arousal,
            control=self.current_mood.control,
            dominant_mood=self.current_mood.dominant_mood,
            intensity=self.current_mood.intensity,
            trigger=trigger
        )
        
        self.mood_history.append(history_entry)
        if len(self.mood_history) > self.max_history:
            self.mood_history.pop(0)

    # Function tools for the agent
    @function_tool
    async def update_mood(self) -> Dict[str, Any]:
        """
        Updates the current mood based on various system states.
        
        Returns:
            Updated mood state information
        """
        async with self._lock:  # Ensure thread safety
            now = datetime.datetime.now()
            if (now - self.last_update_time).total_seconds() < self.update_interval_seconds:
                return self.current_mood.dict()  # Not time to update yet
    
            # Calculate elapsed time factor (adjusts influence strength)
            elapsed = (now - self.last_update_time).total_seconds()
            time_factor = min(1.0, elapsed / (self.update_interval_seconds * 5))  # Full effect after ~5 mins
    
            influences: Dict[str, float] = {}
            target_valence = 0.0
            target_arousal = 0.5
            target_control = 0.0
            total_weight = 0.0
    
            # 1. Influence from recent Emotional State
            if self.emotional_core:
                try:
                    if hasattr(self.emotional_core, 'get_average_recent_state'):
                        avg_emotion = await self.emotional_core.get_average_recent_state(minutes=5)
                        if avg_emotion:
                            emo_valence = avg_emotion.get("valence", 0.0)
                            emo_arousal = avg_emotion.get("arousal", 0.5)
                            
                            # Valence influence
                            weight = self.influence_weights["emotion_valence"]
                            target_valence += emo_valence * weight
                            influences["emotion_valence"] = emo_valence * weight
                            total_weight += weight
    
                            # Arousal influence
                            weight = self.influence_weights["emotion_arousal"]
                            target_arousal += (emo_arousal - 0.5) * weight  # Adjust around neutral 0.5
                            influences["emotion_arousal"] = (emo_arousal - 0.5) * weight
                            total_weight += weight
                            
                            # Extract emotional dominance if available
                            if "dominance" in avg_emotion:
                                emo_dominance = avg_emotion.get("dominance", 0.0)
                                target_control += emo_dominance * weight * 0.7  # Lower impact on control
                except Exception as e:
                    logger.error(f"Error getting emotional influence: {e}")
    
            # 2. Influence from Hormones
            if self.hormone_system:
                try:
                    # Add await here to properly get hormone levels
                    hormone_levels = await self.hormone_system.get_hormone_levels()
                    hormone_influence_valence = 0.0
                    hormone_influence_arousal = 0.0
                    hormone_influence_control = 0.0
                    
                    # Simplified hormone mappings
                    hormone_mappings = {
                        "endoryx": {"valence": 0.4, "arousal": 0.1, "control": 0.1},   # Endorphin -> positive, slight energy
                        "estradyx": {"valence": 0.2, "arousal": -0.1, "control": -0.1}, # Estrogen -> slightly positive, calming
                        "oxytonyx": {"valence": 0.3, "arousal": -0.2, "control": -0.1}, # Oxytocin -> positive, calming
                        "testoryx": {"valence": 0.1, "arousal": 0.3, "control": 0.5},   # Testosterone -> energizing, dominance
                        "melatonyx": {"valence": 0.0, "arousal": -0.5, "control": -0.2}, # Melatonin -> very calming
                        "cortanyx": {"valence": -0.3, "arousal": 0.3, "control": -0.2},  # Cortisol -> negative, energizing, less control
                        "nyxamine": {"valence": 0.3, "arousal": 0.2, "control": 0.3},    # Dopamine -> positive, energizing, control
                        "seranix": {"valence": 0.2, "arousal": -0.1, "control": 0.1},    # Serotonin -> positive, slightly calming
                        "libidyx": {"valence": 0.1, "arousal": 0.5, "control": 0.2},     # Libido -> energizing, slight dominance
                        "serenity_boost": {"valence": 0.3, "arousal": -0.6, "control": -0.1} # Post-gratification -> positive, very calming
                    }
                    
                    # Calculate hormone influence
                    for hormone, mapping in hormone_mappings.items():
                        if hormone in hormone_levels:
                            # Get hormone level (0-1) and normalize around 0.5 baseline
                            level = hormone_levels[hormone].get("value", 0.5) - 0.5
                            
                            # Apply influence based on deviation from baseline
                            hormone_influence_valence += level * mapping["valence"]
                            hormone_influence_arousal += level * mapping["arousal"]
                            hormone_influence_control += level * mapping["control"]
                    
                    # Apply hormone weight
                    weight = self.influence_weights["hormones"]
                    target_valence += hormone_influence_valence * weight
                    target_arousal += hormone_influence_arousal * weight
                    target_control += hormone_influence_control * weight
                    
                    # Store aggregate influence
                    hormone_total = (abs(hormone_influence_valence) + 
                                    abs(hormone_influence_arousal) + 
                                    abs(hormone_influence_control)) / 3
                    influences["hormones"] = hormone_total * weight
                    total_weight += weight
                except Exception as e:
                    logger.error(f"Error getting hormone influence: {e}")
    
            # 3. Influence from Needs
            if self.needs_system:
                try:
                    needs_state = self.needs_system.get_needs_state()
                    if needs_state:
                        # Calculate weighted deficit
                        total_deficit = 0
                        total_importance = 0
                        
                        for need_name, need_data in needs_state.items():
                            deficit = need_data.get('deficit', 0)
                            importance = need_data.get('importance', 0.5)
                            
                            total_deficit += deficit * importance
                            total_importance += importance
                        
                        avg_deficit = total_deficit / total_importance if total_importance > 0 else 0
                        
                        # High deficit -> negative valence, slight arousal increase
                        needs_influence_valence = -avg_deficit * 1.5  # Strong negative impact
                        needs_influence_arousal = avg_deficit * 0.3   # Slight agitation
                        needs_influence_control = -avg_deficit * 0.2  # Reduced sense of control
            
                        # Get weight first before using it
                        weight = self.influence_weights["needs"]
            
                        # Add valence/arousal/control from pleasure deprivation
                        pleasure_state = needs_state.get("pleasure_indulgence", {})
                        pleasure_deficit = pleasure_state.get("deficit", 0.0)
                        pleasure_importance = pleasure_state.get("importance", 0.0)
                        
                        valence_drop = -pleasure_deficit * pleasure_importance * 0.6
                        arousal_boost = pleasure_deficit * 0.4
                        control_bias = pleasure_deficit * 0.2
                        
                        target_valence += valence_drop * weight
                        target_arousal += arousal_boost * weight
                        target_control += control_bias * weight
                        
                        influences["pleasure_deprivation"] = pleasure_deficit * weight
            
                        # Apply needs weight (already defined above)
                        target_valence += needs_influence_valence * weight
                        target_arousal += needs_influence_arousal * weight
                        target_control += needs_influence_control * weight
                        
                        influences["needs"] = (abs(needs_influence_valence) + 
                                            abs(needs_influence_arousal) + 
                                            abs(needs_influence_control)) / 3 * weight
                        total_weight += weight
                except Exception as e:
                    logger.error(f"Error getting needs influence: {e}")
    
            # 4. Influence from Recent Goal Outcomes
            if self.goal_manager:
                try:
                    recent_goals = await self.goal_manager.get_all_goals()
                    # Filter to recently completed/failed goals (last hour)
                    recent_outcomes = []
                    for g in recent_goals:
                        if g.get('completion_time'):
                            try:
                                completion_time = datetime.datetime.fromisoformat(g['completion_time'])
                                if (now - completion_time).total_seconds() < 3600:  # Last hour
                                    recent_outcomes.append(g)
                            except (ValueError, TypeError):
                                pass
                    
                    if recent_outcomes:
                        # Calculate success and failure rates
                        success_rate = sum(1 for g in recent_outcomes if g['status'] == 'completed') / len(recent_outcomes)
                        failure_rate = sum(1 for g in recent_outcomes if g['status'] == 'failed') / len(recent_outcomes)
                        
                        # Success -> positive valence, agency (control)
                        # Failure -> negative valence, less agency
                        goal_influence_valence = (success_rate - failure_rate) * 0.5
                        goal_influence_control = (success_rate - failure_rate) * 0.4
                        
                        # Apply goal weight
                        weight = self.influence_weights["goals"]
                        target_valence += goal_influence_valence * weight
                        target_control += goal_influence_control * weight
                        
                        influences["goals"] = (abs(goal_influence_valence) + 
                                            abs(goal_influence_control)) / 2 * weight
                        total_weight += weight
                except Exception as e:
                    logger.error(f"Error getting goal influence: {e}")
    
            # Calculate weighted average target state (if any influences)
            if total_weight > 0:
                target_valence /= total_weight
                target_arousal = 0.5 + (target_arousal / total_weight)  # Adjust around 0.5 baseline
                target_control /= total_weight
            else:
                # No influences, drift towards neutral
                target_valence = 0.0
                target_arousal = 0.5
                target_control = 0.0
    
            # Apply inertia and update mood dimensions
            # Adjust update strength by time factor
            update_strength = 1.0 - (self.inertia * (1.0 - time_factor))  # Less inertia if more time passed
    
            new_valence = self.current_mood.valence * (1.0 - update_strength) + target_valence * update_strength
            new_arousal = self.current_mood.arousal * (1.0 - update_strength) + target_arousal * update_strength
            new_control = self.current_mood.control * (1.0 - update_strength) + target_control * update_strength
    
            # Clamp values
            self.current_mood.valence = max(-1.0, min(1.0, new_valence))
            self.current_mood.arousal = max(0.0, min(1.0, new_arousal))
            self.current_mood.control = max(-1.0, min(1.0, new_control))
    
            # Derive dominant mood label and intensity
            self.current_mood.dominant_mood = self._get_dominant_mood_label(
                self.current_mood.valence, self.current_mood.arousal, self.current_mood.control
            )
            
            # Intensity based on distance from neutral origin (0, 0.5, 0)
            intensity = math.sqrt(
                self.current_mood.valence**2 + 
                ((self.current_mood.arousal - 0.5)*2)**2 + 
                self.current_mood.control**2
            ) / math.sqrt(1**2 + 1**2 + 1**2)  # Normalize by max possible distance
            
            self.current_mood.intensity = min(1.0, intensity)
    
            # Update timestamp and influences
            self.current_mood.last_updated = now
            self.current_mood.influences = {k: v for k, v in influences.items() if abs(v) > 0.01}  # Store significant influences
            self.last_update_time = now
    
            # Add to history
            self._add_to_history("periodic_update")
    
            logger.info(f"Mood updated: {self.current_mood.dominant_mood} (V:{self.current_mood.valence:.2f} A:{self.current_mood.arousal:.2f} C:{self.current_mood.control:.2f})")
            return self.current_mood.dict()

    @function_tool
    async def get_current_mood(self) -> MoodState:
        """Returns the current mood state, updating if needed."""
        now = datetime.datetime.now()
        
        # If the mood is stale, update it
        if (now - self.last_update_time).total_seconds() > self.update_interval_seconds:
            await self.update_mood()
            
        return self.current_mood
        
    @function_tool
    async def modify_mood(
        self, 
        valence_change: Optional[float] = None,
        arousal_change: Optional[float] = None,
        control_change: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Manually modify the current mood (for testing or external influences).
        
        Args:
            valence_change: Change to valence (pleasantness) dimension (-1.0 to 1.0)
            arousal_change: Change to arousal (energy) dimension (-1.0 to 1.0)
            control_change: Change to control (dominance) dimension (-1.0 to 1.0)
            reason: Reason for the mood modification
            
        Returns:
            Updated mood state
        """
        # Set default values inside the function body
        valence_change = 0 if valence_change is None else valence_change
        arousal_change = 0 if arousal_change is None else arousal_change
        control_change = 0 if control_change is None else control_change
        reason = "manual_adjustment" if reason is None else reason
        
        async with self._lock:
            # Get current mood to ensure it's up to date
            await self.get_current_mood()
            
            # Apply changes
            self.current_mood.valence = max(-1.0, min(1.0, self.current_mood.valence + valence_change))
            self.current_mood.arousal = max(0.0, min(1.0, self.current_mood.arousal + arousal_change))
            self.current_mood.control = max(-1.0, min(1.0, self.current_mood.control + control_change))
            
            # Update derived properties
            self.current_mood.dominant_mood = self._get_dominant_mood_label(
                self.current_mood.valence, self.current_mood.arousal, self.current_mood.control
            )
            
            # Update intensity
            intensity = math.sqrt(
                self.current_mood.valence**2 + 
                ((self.current_mood.arousal - 0.5)*2)**2 + 
                self.current_mood.control**2
            ) / math.sqrt(1**2 + 1**2 + 1**2)
            
            self.current_mood.intensity = min(1.0, intensity)
            self.current_mood.last_updated = datetime.datetime.now()
            
            # Add to history
            self._add_to_history(f"manual:{reason}")
            
            return self.current_mood.dict()
    
    @function_tool
    async def handle_significant_event(
        self,
        event_type: str, 
        intensity: float, 
        valence: float,
        arousal_change: Optional[float] = None,
        control_change: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Handle a significant event that should influence mood.
        
        Args:
            event_type: Type of event (e.g., "user_feedback", "system_error")
            intensity: How strong the event's influence should be (0-1)
            valence: The positive/negative nature of the event (-1 to 1)
            arousal_change: Optional specific change to arousal
            control_change: Optional specific change to sense of control
            
        Returns:
            Updated mood state
        """
        async with self._lock:
            # Calculate changes based on event properties
            weight = min(1.0, intensity * self.influence_weights.get("external_events", 0.2))
            
            # Default arousal and control changes if not specified
            if arousal_change is None:
                # High intensity events increase arousal for both positive/negative events
                arousal_change = (intensity - 0.5) * 0.3
                
            if control_change is None:
                # Positive events increase control, negative decrease it
                control_change = valence * intensity * 0.2
            
            # Apply changes
            valence_change = valence * weight
            arousal_change = arousal_change * weight
            control_change = control_change * weight
            
            # Update mood
            result = await self.modify_mood(
                valence_change=valence_change,
                arousal_change=arousal_change,
                control_change=control_change,
                reason=f"event:{event_type}"
            )
            
            return result
    
    @function_tool
    async def get_mood_history(
        self,
        hours: Optional[int] = None,
        include_details: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get mood history for the specified time period.
        
        Args:
            hours: How many hours of history to retrieve
            include_details: Whether to include full details or just summaries
            
        Returns:
            List of mood states with timestamps
        """
        # Handle default values inside the function
        hours = 24 if hours is None else hours
        include_details = False if include_details is None else include_details
        
        async with self._lock:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            
            # Filter history to requested time period
            filtered_history = [
                entry for entry in self.mood_history
                if entry.timestamp >= cutoff_time
            ]
            
            # Format based on detail level
            if include_details:
                return [entry.dict() for entry in filtered_history]
            else:
                return [
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "mood": entry.dominant_mood,
                        "valence": round(entry.valence, 2),
                        "arousal": round(entry.arousal, 2),
                        "intensity": round(entry.intensity, 2),
                        "trigger": entry.trigger
                    }
                    for entry in filtered_history
                ]
    
    # Public API methods - directly call the function tools when used externally
    async def run_update_mood(self) -> MoodState:
        """Public API to update the mood state."""
        mood_data = await self.update_mood()
        if isinstance(mood_data, dict):
            return MoodState(**mood_data)
        return self.current_mood

    async def run_get_current_mood(self) -> MoodState:
        """Public API to get the current mood state."""
        return await self.get_current_mood()

    async def run_modify_mood(self, valence_change: float = 0, 
                          arousal_change: float = 0, 
                          control_change: float = 0,
                          reason: str = "manual_adjustment") -> Dict[str, Any]:
        """Public API to modify the mood state."""
        return await self.modify_mood(
            valence_change=valence_change,
            arousal_change=arousal_change,
            control_change=control_change,
            reason=reason
        )

    async def run_handle_significant_event(self, event_type: str, 
                                      intensity: float, 
                                      valence: float,
                                      arousal_change: Optional[float] = None,
                                      control_change: Optional[float] = None) -> Dict[str, Any]:
        """Public API to handle a significant event."""
        return await self.handle_significant_event(
            event_type=event_type,
            intensity=intensity,
            valence=valence,
            arousal_change=arousal_change,
            control_change=control_change
        )

    async def run_get_mood_history(self, hours: int = 24, 
                                include_details: bool = False) -> List[Dict[str, Any]]:
        """Public API to get mood history."""
        return await self.get_mood_history(
            hours=hours,
            include_details=include_details
        )
