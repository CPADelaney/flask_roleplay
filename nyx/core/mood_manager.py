# nyx/core/mood_manager.py

import logging
import datetime
import math
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

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

class MoodManager:
    """Manages Nyx's mid-term affective state (mood)."""

    def __init__(self, emotional_core=None, hormone_system=None, needs_system=None, goal_manager=None):
        self.emotional_core = emotional_core
        self.hormone_system = hormone_system
        self.needs_system = needs_system
        self.goal_manager = goal_manager # To check goal success/failure

        self.current_mood = MoodState()
        self.mood_history: List[MoodState] = []
        self.max_history = 100

        # Configuration
        self.inertia = 0.90 # How much the previous mood state persists (0-1)
        self.update_interval_seconds = 60 # How often to update mood
        self.last_update_time = datetime.datetime.now()

        # Weights for different influences
        self.influence_weights = {
            "emotion_valence": 0.4,
            "emotion_arousal": 0.3,
            "hormones": 0.15,
            "needs": 0.1,
            "goals": 0.05,
        }

        logger.info("MoodManager initialized.")

    async def update_mood(self) -> MoodState:
        """Updates the current mood based on various system states."""
        now = datetime.datetime.now()
        if (now - self.last_update_time).total_seconds() < self.update_interval_seconds:
            return self.current_mood # Not time to update yet

        # Calculate elapsed time factor (adjusts influence strength)
        elapsed = (now - self.last_update_time).total_seconds()
        time_factor = min(1.0, elapsed / (self.update_interval_seconds * 5)) # Full effect after ~5 mins

        influences: Dict[str, float] = {}
        target_valence = 0.0
        target_arousal = 0.5
        target_control = 0.0
        total_weight = 0.0

        # 1. Influence from recent Emotional State
        if self.emotional_core and hasattr(self.emotional_core, 'get_average_recent_state'):
            # Assumes EmotionalCore can provide avg valence/arousal over last N minutes
            avg_emotion = await self.emotional_core.get_average_recent_state(minutes=5)
            if avg_emotion:
                emo_valence = avg_emotion.get("valence", 0.0)
                emo_arousal = avg_emotion.get("arousal", 0.5)
                weight = self.influence_weights["emotion_valence"]
                target_valence += emo_valence * weight
                influences["emotion_valence"] = emo_valence * weight
                total_weight += weight

                weight = self.influence_weights["emotion_arousal"]
                target_arousal += (emo_arousal - 0.5) * weight # Adjust around neutral 0.5
                influences["emotion_arousal"] = (emo_arousal - 0.5) * weight
                total_weight += weight
                # Control dimension could be influenced by dominant emotions like Anger/Dominance vs Fear/Sadness
                # Simplified for now

        # 2. Influence from Hormones
        if self.hormone_system:
            hormone_levels = self.hormone_system.get_hormone_levels()
            hormone_influence_valence = 0.0
            hormone_influence_arousal = 0.0
            hormone_influence_control = 0.0
            # Simplified mapping:
            hormone_influence_valence += (hormone_levels.get("endoryx", {}).get("value", 0.5) - 0.5) * 0.4 # Endorphin -> positive
            hormone_influence_valence += (hormone_levels.get("estradyx", {}).get("value", 0.5) - 0.5) * 0.2 # Estrogen -> slightly positive
            hormone_influence_valence += (hormone_levels.get("oxytonyx", {}).get("value", 0.5) - 0.5) * 0.3 # Oxytocin -> positive
            hormone_influence_arousal += (hormone_levels.get("testoryx", {}).get("value", 0.5) - 0.5) * 0.3 # Testosterone -> arousal
            hormone_influence_arousal -= (hormone_levels.get("melatonyx", {}).get("value", 0.3) - 0.3) * 0.5 # Melatonin -> lower arousal
            hormone_influence_control += (hormone_levels.get("testoryx", {}).get("value", 0.5) - 0.5) * 0.5 # Testosterone -> dominance

            weight = self.influence_weights["hormones"]
            target_valence += hormone_influence_valence * weight
            target_arousal += hormone_influence_arousal * weight
            target_control += hormone_influence_control * weight
            influences["hormones"] = (hormone_influence_valence + hormone_influence_arousal + hormone_influence_control) * weight / 3
            total_weight += weight

        # 3. Influence from Needs
        if self.needs_system:
            needs_state = self.needs_system.get_needs_state()
            total_deficit = sum(n['deficit'] * n['importance'] for n in needs_state.values())
            avg_deficit = total_deficit / len(needs_state) if needs_state else 0
            # High deficit -> negative valence, maybe slight arousal increase (agitation)
            needs_influence_valence = -avg_deficit * 1.5 # Deficit strongly impacts mood negatively
            needs_influence_arousal = avg_deficit * 0.3 # Slight agitation from unmet needs

            weight = self.influence_weights["needs"]
            target_valence += needs_influence_valence * weight
            target_arousal += needs_influence_arousal * weight
            influences["needs"] = (needs_influence_valence + needs_influence_arousal) * weight / 2
            total_weight += weight

        # 4. Influence from Recent Goal Outcomes
        if self.goal_manager:
            recent_goals = await self.goal_manager.get_all_goals() # Get summary
            recent_outcomes = [g for g in recent_goals if g.get('completion_time') and (now - datetime.datetime.fromisoformat(g['completion_time'])).total_seconds() < 3600] # Last hour
            if recent_outcomes:
                 success_rate = sum(1 for g in recent_outcomes if g['status'] == 'completed') / len(recent_outcomes)
                 failure_rate = sum(1 for g in recent_outcomes if g['status'] == 'failed') / len(recent_outcomes)
                 # Success -> positive valence, agency (control)
                 # Failure -> negative valence, less agency
                 goal_influence_valence = (success_rate - failure_rate) * 0.5
                 goal_influence_control = (success_rate - failure_rate) * 0.4 # Link agency/control to goal success

                 weight = self.influence_weights["goals"]
                 target_valence += goal_influence_valence * weight
                 target_control += goal_influence_control * weight
                 influences["goals"] = (goal_influence_valence + goal_influence_control) * weight / 2
                 total_weight += weight

        # Calculate weighted average target state (if any influences)
        if total_weight > 0:
             target_valence /= total_weight
             # Adjust target arousal around 0.5 baseline
             target_arousal = 0.5 + (target_arousal / total_weight)
             target_control /= total_weight
        else: # No influences, drift towards neutral
             target_valence = 0.0
             target_arousal = 0.5
             target_control = 0.0

        # Apply inertia and update mood dimensions
        # Adjust update speed by time factor
        update_strength = 1.0 - (self.inertia * (1.0 - time_factor)) # Less inertia if more time passed

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
            self.current_mood.valence**2 + \
            ((self.current_mood.arousal - 0.5)*2)**2 + \
            self.current_mood.control**2
        ) / math.sqrt(1**2 + 1**2 + 1**2) # Normalize by max possible distance
        self.current_mood.intensity = min(1.0, intensity)

        # Update timestamp and influences
        self.current_mood.last_updated = now
        self.current_mood.influences = {k: v for k,v in influences.items() if abs(v)>0.01} # Store significant influences
        self.last_update_time = now

        # Add to history
        self.mood_history.append(self.current_mood.copy())
        if len(self.mood_history) > self.max_history:
            self.mood_history.pop(0)

        logger.info(f"Mood updated: {self.current_mood.dominant_mood} (V:{self.current_mood.valence:.2f} A:{self.current_mood.arousal:.2f} C:{self.current_mood.control:.2f})")
        return self.current_mood

    def get_current_mood(self) -> MoodState:
        """Returns the current mood state."""
        # Run update first if needed (optional, depends if external loop calls update)
        # asyncio.run(self.update_mood()) # Careful with running async in sync method
        return self.current_mood

    def _get_dominant_mood_label(self, v: float, a: float, c: float) -> str:
        """Maps mood dimensions to a descriptive label."""
        # Simple mapping based on quadrants/octants of V-A-C space
        if a > 0.7: # High Arousal
            if v > 0.3: return "Excited" if c > 0.3 else ("Elated" if c < -0.3 else "Enthusiastic")
            if v < -0.3: return "Anxious" if c > 0.3 else ("Stressed" if c < -0.3 else "Tense")
            return "Alert" if c > 0.3 else ("Agitated" if c < -0.3 else "Focused")
        elif a < 0.3: # Low Arousal
             if v > 0.3: return "Calm" if c > 0.3 else ("Relaxed" if c < -0.3 else "Content")
             if v < -0.3: return "Depressed" if c > 0.3 else ("Sad" if c < -0.3 else "Bored")
             return "Passive" if c > 0.3 else ("Lethargic" if c < -0.3 else "Drowsy")
        else: # Mid Arousal
             if v > 0.5: return "Happy" if c > 0.3 else ("Pleased" if c < -0.3 else "Glad")
             if v < -0.5: return "Unhappy" if c > 0.3 else ("Displeased" if c < -0.3 else "Gloomy")
             return "Neutral" if abs(c)<0.3 else ("Confident" if c>0 else "Reserved")
