from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import random
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import math
import re

class ReinforcementAgent(BaseModel):
    """Agent responsible for reinforcement learning and behavior optimization"""
    id: str = Field(default_factory=lambda: f"reinforce_{random.randint(1000, 9999)}")
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    discount_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    exploration_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # State tracking
    state_history: List[Dict[str, Any]] = []
    action_history: List[Dict[str, Any]] = []
    reward_history: List[Dict[str, Any]] = []
    
    # Session management
    last_interaction_time: Optional[datetime] = None
    session_start_time: Optional[datetime] = None
    current_session_id: Optional[str] = None
    session_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    player_desperation: float = Field(default=0.0, ge=0.0, le=1.0)
    intensity_ramp_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    desperation_decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    
    # Knowledge bases
    punishment_ideas: Dict[str, List[Dict[str, Any]]] = {}
    reward_ideas: Dict[str, List[Dict[str, Any]]] = {}
    narrative_patterns: Dict[str, List[Dict[str, Any]]] = {}
    character_archetypes: Dict[str, List[Dict[str, Any]]] = {}
    
    # Performance metrics
    creativity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    narrative_score: float = Field(default=0.5, ge=0.0, le=1.0)
    character_score: float = Field(default=0.5, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Learning parameters
    state_space: Dict[str, Any] = {}
    action_space: Dict[str, Any] = {}
    q_table: Dict[str, Dict[str, float]] = {}
    
    # Add new tracking metrics
    addiction_level: float = Field(default=0.0, ge=0.0, le=1.0)
    obsession_level: float = Field(default=0.0, ge=0.0, le=1.0)
    dependency_metrics: Dict[str, float] = Field(default_factory=lambda: {
        "emotional_dependency": 0.0,
        "psychological_dependency": 0.0,
        "behavioral_dependency": 0.0,
        "reward_conditioning": 0.0,
        "withdrawal_sensitivity": 0.0
    })
    
    # Add dependency progression tracking
    dependency_stages: Dict[str, bool] = Field(default_factory=lambda: {
        "initial_hook": False,
        "early_dependency": False,
        "deep_conditioning": False,
        "psychological_binding": False,
        "complete_addiction": False
    })
    
    # Add manipulation strategy tracking
    manipulation_success: Dict[str, float] = Field(default_factory=lambda: {
        "reward_reinforcement": 0.0,
        "psychological_hooks": 0.0,
        "emotional_triggers": 0.0,
        "behavioral_patterns": 0.0
    })
    
    # Add psychological manipulation metrics
    psychological_metrics: Dict[str, float] = Field(default_factory=lambda: {
        "gaslighting_effectiveness": 0.0,
        "cognitive_dissonance": 0.0,
        "ego_destruction": 0.0,
        "reality_distortion": 0.0,
        "isolation_progress": 0.0
    })
    
    # Add reward conditioning metrics
    conditioning_metrics: Dict[str, float] = Field(default_factory=lambda: {
        "pleasure_response": 0.0,
        "anticipation_level": 0.0,
        "reward_craving": 0.0,
        "pavlovian_response": 0.0,
        "behavioral_reinforcement": 0.0
    })
    
    # Add emotional dependency metrics
    emotional_metrics: Dict[str, float] = Field(default_factory=lambda: {
        "attachment_strength": 0.0,
        "validation_need": 0.0,
        "separation_anxiety": 0.0,
        "emotional_reliance": 0.0,
        "identity_fusion": 0.0
    })
    
    # Add advanced manipulation metrics
    manipulation_layers: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "primary": {
            "self_doubt": 0.0,
            "emotional_vulnerability": 0.0,
            "dependency_reinforcement": 0.0
        },
        "secondary": {
            "identity_erosion": 0.0,
            "reality_questioning": 0.0,
            "attachment_deepening": 0.0
        },
        "compound": {
            "psychological_anchoring": 0.0,
            "behavioral_conditioning": 0.0,
            "emotional_binding": 0.0
        }
    })

    # Enhanced reward scaling system
    reward_scaling: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "base_multipliers": {
            "pleasure": 1.0,
            "validation": 1.0,
            "attachment": 1.0
        },
        "compound_effects": {
            "consecutive_rewards": 1.0,
            "intensity_boost": 1.0,
            "psychological_impact": 1.0
        },
        "thresholds": {
            "pleasure_threshold": 0.7,
            "validation_threshold": 0.6,
            "attachment_threshold": 0.8
        }
    })
    
    # Add NPC interaction capabilities
    npc_profile: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "Nyx",
        "appearance": {
            "height": 175,  # cm
            "build": "athletic",
            "hair": "raven black",
            "eyes": "deep violet",
            "features": ["elegant", "striking", "mysterious"],
            "style": "sophisticated dark"
        },
        "personality": {
            "core_traits": ["manipulative", "seductive", "intelligent", "dominant"],
            "adaptable_traits": ["playful", "stern", "nurturing", "cruel"],
            "current_mood": "neutral",
            "power_dynamic": 0.8  # 0-1 scale of dominance
        },
        "abilities": {
            "physical": ["graceful", "agile", "strong"],
            "mental": ["perceptive", "strategic", "persuasive"],
            "special": ["emotional manipulation", "psychological insight"]
        },
        "relationships": {},  # Track relationships with other characters
        "status": {
            "is_active": False,
            "current_scene": None,
            "current_target": None,
            "interaction_history": []
        }
    })

    # Enhanced manipulation patterns
    advanced_manipulation: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "psychological": {
            "gaslighting": {
                "doubt_induction": 0.0,
                "reality_distortion": 0.0,
                "memory_manipulation": 0.0,
                "emotional_confusion": 0.0
            },
            "conditioning": {
                "reward_association": 0.0,
                "punishment_association": 0.0,
                "intermittent_reinforcement": 0.0,
                "behavioral_shaping": 0.0
            },
            "cognitive_restructuring": {
                "belief_modification": 0.0,
                "value_shifting": 0.0,
                "identity_erosion": 0.0,
                "worldview_alteration": 0.0
            }
        },
        "emotional": {
            "trauma_bonding": {
                "intensity": 0.0,
                "attachment_depth": 0.0,
                "dependency_level": 0.0,
                "emotional_anchoring": 0.0
            },
            "emotional_exploitation": {
                "vulnerability_targeting": 0.0,
                "emotional_triggering": 0.0,
                "empathy_manipulation": 0.0,
                "guilt_leveraging": 0.0
            },
            "affective_conditioning": {
                "pleasure_association": 0.0,
                "pain_association": 0.0,
                "emotional_anchoring": 0.0,
                "mood_manipulation": 0.0
            }
        },
        "behavioral": {
            "operant_conditioning": {
                "reward_timing": 0.0,
                "punishment_timing": 0.0,
                "reinforcement_schedule": 0.0,
                "behavioral_chaining": 0.0
            },
            "social_conditioning": {
                "isolation_inducement": 0.0,
                "dependency_creation": 0.0,
                "social_proof": 0.0,
                "authority_leveraging": 0.0
            },
            "habit_formation": {
                "routine_building": 0.0,
                "trigger_establishment": 0.0,
                "compulsion_development": 0.0,
                "behavioral_lock-in": 0.0
            }
        }
    })

    # Enhanced reward scaling
    advanced_reward_scaling: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "psychological": {
            "base_multiplier": 1.0,
            "cognitive_impact": 0.0,
            "mental_state": 0.0,
            "psychological_investment": 0.0,
            "thresholds": {
                "cognitive": 0.7,
                "mental": 0.6,
                "investment": 0.8
            }
        },
        "emotional": {
            "base_multiplier": 1.0,
            "emotional_intensity": 0.0,
            "affective_bond": 0.0,
            "emotional_dependency": 0.0,
            "thresholds": {
                "intensity": 0.7,
                "bond": 0.8,
                "dependency": 0.9
            }
        },
        "behavioral": {
            "base_multiplier": 1.0,
            "action_frequency": 0.0,
            "response_consistency": 0.0,
            "behavioral_commitment": 0.0,
            "thresholds": {
                "frequency": 0.6,
                "consistency": 0.7,
                "commitment": 0.8
            }
        },
        "compound_effects": {
            "psychological_emotional": 1.0,
            "emotional_behavioral": 1.0,
            "psychological_behavioral": 1.0,
            "total_synergy": 1.0
        }
    })
    
    async def process_interaction(self, event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an interaction and update learning state"""
        # Update session state
        self._update_session_state(event)
        
        # Extract current state
        state = self._extract_state(event, context)
        
        # Calculate rewards
        rewards = self._calculate_rewards(event, context)
        
        # Update learning state
        self._update_q_values(state, rewards)
        
        # Store experience
        self._store_experience(state, rewards)
        
        # Update performance metrics
        self._update_performance_metrics(rewards)
        
        # Generate insights
        insights = self._generate_learning_insights()
        
        # Generate recommended actions with intensity consideration
        actions = self._generate_recommended_actions()
        
        return {
            "state": state,
            "rewards": rewards,
            "insights": insights,
            "recommended_actions": actions,
            "session_state": {
                "is_fresh_session": self._is_fresh_session(),
                "current_intensity": self.session_intensity,
                "player_desperation": self.player_desperation,
                "session_duration": self._get_session_duration()
            }
        }
    
    def _update_session_state(self, event: Dict[str, Any]):
        """Update session state based on interaction timing"""
        current_time = datetime.now()
        
        # Check if this is a fresh session
        if self._is_fresh_session():
            self._start_new_session(current_time)
        else:
            self._update_existing_session(current_time)
        
        # Update last interaction time
        self.last_interaction_time = current_time
        
        # Update player desperation based on response
        self._update_player_desperation(event)
        
        # Update session intensity
        self._update_session_intensity()
    
    def _is_fresh_session(self) -> bool:
        """Determine if this interaction starts a fresh session"""
        if not self.last_interaction_time:
            return True
            
        # Calculate time since last interaction
        time_since_last = datetime.now() - self.last_interaction_time
        
        # Consider it a fresh session if:
        # 1. More than 30 minutes have passed
        # 2. Or more than 5 minutes with high desperation
        return (time_since_last.total_seconds() > 1800 or  # 30 minutes
                (time_since_last.total_seconds() > 300 and self.player_desperation > 0.8))  # 5 minutes with high desperation
    
    def _start_new_session(self, current_time: datetime):
        """Initialize a new session"""
        self.session_start_time = current_time
        self.current_session_id = f"session_{random.randint(1000, 9999)}"
        self.session_intensity = 0.0  # Start slow
        self.player_desperation = 0.0  # Reset desperation
    
    def _update_existing_session(self, current_time: datetime):
        """Update an existing session"""
        if not self.session_start_time:
            self._start_new_session(current_time)
            return
            
        # Calculate session duration
        session_duration = current_time - self.session_start_time
        
        # Adjust intensity ramp rate based on session duration
        if session_duration.total_seconds() > 3600:  # After 1 hour
            self.intensity_ramp_rate = min(self.intensity_ramp_rate * 1.2, 0.3)
        elif session_duration.total_seconds() > 7200:  # After 2 hours
            self.intensity_ramp_rate = min(self.intensity_ramp_rate * 1.5, 0.5)
    
    def _update_player_desperation(self, event: Dict[str, Any]):
        """Update player desperation based on interaction"""
        # Extract emotional indicators from event with enhanced analysis
        emotional_indicators = self._extract_emotional_indicators(event)
        
        # Calculate desperation increase
        desperation_increase = self._calculate_desperation_increase(emotional_indicators)
        
        # Apply increase with decay
        self.player_desperation = min(
            1.0,
            self.player_desperation + desperation_increase
        )
        
        # Apply natural decay
        self.player_desperation = max(
            0.0,
            self.player_desperation - self.desperation_decay_rate
        )
    
    def _extract_emotional_indicators(self, event: Dict[str, Any]) -> Dict[str, float]:
        """Extract emotional indicators from event with enhanced analysis"""
        content = event.get("content", "")
        
        # Define comprehensive emotional indicators and their weights
        indicators = {
            "submission": {
                "words": ["please", "mistress", "goddess", "yes", "begging", "beg", "kneel", "submit", "yours", "owned", "worship", "devoted", "serve", "obey", "yield"],
                "phrases": ["i'll do anything", "i'm yours", "at your mercy", "under your control", "completely yours"],
                "weight": 0.2
            },
            "desperation": {
                "words": ["need", "want", "crave", "desperate", "aching", "dying", "can't", "can't take", "too much", "overwhelmed", "starving", "thirsting", "yearning", "longing", "pining"],
                "phrases": ["i need it", "i can't take it", "too much", "please let me", "i'm begging", "i'll do anything"],
                "weight": 0.2
            },
            "arousal": {
                "words": ["hard", "wet", "throbbing", "pulsing", "moaning", "leaking", "dripping", "twitching", "trembling", "shaking", "quivering", "panting", "gasping", "arching", "squirming"],
                "phrases": ["so turned on", "so wet", "so hard", "can't stop", "feeling it", "getting close"],
                "weight": 0.15
            },
            "frustration": {
                "words": ["frustrated", "teased", "tortured", "denied", "edge", "edging", "almost", "close", "not yet", "waiting", "agitated", "restless", "impatient", "desperate", "aching"],
                "phrases": ["so close", "almost there", "can't take it", "need release", "too much", "please let me"],
                "weight": 0.15
            },
            "embarrassment": {
                "words": ["embarrassed", "ashamed", "blushing", "red", "shy", "shame", "humiliated", "exposed", "vulnerable", "weak", "flustered", "mortified", "self-conscious", "nervous", "anxious"],
                "phrases": ["so embarrassed", "can't believe", "feel so exposed", "so vulnerable", "so ashamed"],
                "weight": 0.1
            },
            "devotion": {
                "words": ["love", "adore", "worship", "devoted", "dedicated", "loyal", "faithful", "committed", "obsessed", "infatuated"],
                "phrases": ["i love you", "i adore you", "i worship you", "i'm yours", "forever yours"],
                "weight": 0.1
            },
            "anxiety": {
                "words": ["anxious", "nervous", "worried", "afraid", "scared", "fearful", "terrified", "panicked", "overwhelmed", "stressed"],
                "phrases": ["so nervous", "so anxious", "so worried", "can't handle", "too much"],
                "weight": 0.1
            }
        }
        
        # Calculate scores for each indicator
        scores = {}
        for indicator, config in indicators.items():
            # Word-based scoring
            word_score = 0.0
            for word in config["words"]:
                if word in content.lower():
                    word_score += 1.0
            
            # Phrase-based scoring (weighted higher)
            phrase_score = 0.0
            for phrase in config["phrases"]:
                if phrase in content.lower():
                    phrase_score += 2.0
            
            # Combine scores with weights
            total_score = (word_score / len(config["words"]) + phrase_score / len(config["phrases"])) * config["weight"]
            scores[indicator] = min(total_score, 1.0)
        
        # Add emotional intensity analysis
        intensity_score = self._analyze_emotional_intensity(content)
        scores["intensity"] = intensity_score
        
        # Add emotional complexity analysis
        complexity_score = self._analyze_emotional_complexity(scores)
        scores["complexity"] = complexity_score
        
        return scores
    
    def _analyze_emotional_intensity(self, content: str) -> float:
        """Analyze the intensity of emotional expression"""
        # Define intensity indicators
        intensity_indicators = {
            "exclamation": ["!", "!!", "!!!"],
            "capitalization": ["ALL CAPS", "Mixed Case"],
            "repetition": ["please please", "yes yes", "no no"],
            "emphasis": ["so much", "really", "absolutely", "completely", "totally"],
            "urgency": ["now", "right now", "immediately", "quick", "fast"]
        }
        
        intensity_score = 0.0
        total_indicators = sum(len(indicators) for indicators in intensity_indicators.values())
        
        # Check for exclamation marks
        for exclamation in intensity_indicators["exclamation"]:
            if exclamation in content:
                intensity_score += 1.0
        
        # Check for capitalization patterns
        if content.isupper():
            intensity_score += 2.0
        elif any(word.isupper() for word in content.split()):
            intensity_score += 1.0
        
        # Check for repetition
        words = content.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                intensity_score += 1.0
        
        # Check for emphasis words
        for emphasis in intensity_indicators["emphasis"]:
            if emphasis in content.lower():
                intensity_score += 1.0
        
        # Check for urgency
        for urgency in intensity_indicators["urgency"]:
            if urgency in content.lower():
                intensity_score += 1.0
        
        return min(intensity_score / total_indicators, 1.0)
    
    def _analyze_emotional_complexity(self, emotional_scores: Dict[str, float]) -> float:
        """Analyze the complexity of emotional expression"""
        # Count number of significant emotions
        significant_emotions = sum(1 for score in emotional_scores.values() if score > 0.3)
        
        # Calculate emotional diversity
        total_score = sum(emotional_scores.values())
        if total_score == 0:
            return 0.0
        
        # Shannon's diversity index for emotions
        proportions = [score/total_score for score in emotional_scores.values() if score > 0]
        diversity = -sum(p * np.log2(p) for p in proportions)
        
        # Combine metrics
        complexity_score = (
            (significant_emotions / len(emotional_scores)) * 0.4 +  # Number of emotions
            (diversity / np.log2(len(emotional_scores))) * 0.6      # Emotional diversity
        )
        
        return min(complexity_score, 1.0)
    
    def _calculate_desperation_increase(self, indicators: Dict[str, float]) -> float:
        """Calculate increase in player desperation based on indicators"""
        # Weight the indicators
        weighted_sum = sum(indicators.values())
        
        # Apply non-linear scaling for more dramatic increases at higher levels
        if self.player_desperation > 0.7:
            return weighted_sum * 1.5  # Faster increase when already desperate
        elif self.player_desperation > 0.4:
            return weighted_sum * 1.2  # Moderate increase when moderately desperate
        else:
            return weighted_sum  # Normal increase when starting
    
    def _update_session_intensity(self):
        """Update session intensity based on player state"""
        # Base intensity increase
        intensity_increase = self.intensity_ramp_rate
        
        # Adjust based on player desperation
        if self.player_desperation > 0.8:
            intensity_increase *= 1.5  # Faster ramp when very desperate
        elif self.player_desperation > 0.5:
            intensity_increase *= 1.2  # Moderate ramp when moderately desperate
        
        # Apply intensity increase
        self.session_intensity = min(1.0, self.session_intensity + intensity_increase)
    
    def _get_session_duration(self) -> float:
        """Get current session duration in seconds"""
        if not self.session_start_time:
            return 0.0
        return (datetime.now() - self.session_start_time).total_seconds()
    
    def _generate_recommended_actions(self) -> List[Dict[str, Any]]:
        """Generate recommended actions with intensity consideration"""
        actions = []
        
        # Get base actions
        base_actions = self._get_base_actions()
        
        # Adjust actions based on session state
        for action in base_actions:
            adjusted_action = self._adjust_action_intensity(action)
            actions.append(adjusted_action)
        
        return actions
    
    def _get_base_actions(self) -> List[Dict[str, Any]]:
        """Get base set of actions without intensity adjustment"""
        return [
            {
                "type": "tease",
                "intensity": 0.5,
                "content": "A gentle reminder of who's in control..."
            },
            {
                "type": "command",
                "intensity": 0.7,
                "content": "A firm directive to follow..."
            },
            {
                "type": "reward",
                "intensity": 0.6,
                "content": "A tempting reward for good behavior..."
            },
            {
                "type": "punishment",
                "intensity": 0.8,
                "content": "A consequence for disobedience..."
            }
        ]
    
    def _adjust_action_intensity(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust action intensity based on session state"""
        adjusted_action = action.copy()
        
        # Scale intensity based on session intensity
        base_intensity = action.get("intensity", 0.5)
        adjusted_intensity = base_intensity * self.session_intensity
        
        # Further adjust based on player desperation
        if self.player_desperation > 0.8:
            adjusted_intensity *= 1.3  # More intense when very desperate
        elif self.player_desperation > 0.5:
            adjusted_intensity *= 1.1  # Slightly more intense when moderately desperate
        
        # Cap intensity at 1.0
        adjusted_action["intensity"] = min(1.0, adjusted_intensity)
        
        # Adjust content based on intensity
        adjusted_action["content"] = self._adjust_content_intensity(
            action["content"],
            adjusted_action["intensity"]
        )
        
        return adjusted_action
    
    def _adjust_content_intensity(self, content: str, intensity: float) -> str:
        """Adjust content wording based on intensity level"""
        # Define intensity modifiers
        modifiers = {
            "gentle": ["soft", "gentle", "sweet", "tender"],
            "moderate": ["firm", "clear", "direct", "steady"],
            "intense": ["harsh", "strict", "demanding", "forceful"],
            "extreme": ["brutal", "merciless", "relentless", "uncompromising"]
        }
        
        # Select appropriate modifier based on intensity
        if intensity < 0.25:
            modifier = random.choice(modifiers["gentle"])
        elif intensity < 0.5:
            modifier = random.choice(modifiers["moderate"])
        elif intensity < 0.75:
            modifier = random.choice(modifiers["intense"])
        else:
            modifier = random.choice(modifiers["extreme"])
        
        # Apply modifier to content
        return f"{modifier.capitalize()} {content.lower()}"
    
    def _extract_state(self, event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant state information from the event and context"""
        return {
            "emotional_state": context.get("emotional_state", {}),
            "scene_context": context.get("scene_context", {}),
            "player_profile": context.get("player_profile", {}),
            "interaction_type": event.get("type", "unknown"),
            "content": event.get("content", ""),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_rewards(self, event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate rewards for different aspects of Nyx's behavior"""
        rewards = {
            "creativity": self._reward_creativity(event, context),
            "narrative": self._reward_narrative(event, context),
            "character": self._reward_character(event, context),
            "consistency": self._reward_consistency(event, context)
        }
        
        # Update performance metrics
        self._update_performance_metrics(rewards)
        
        return rewards
    
    def _reward_creativity(self, event: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Reward creative punishments, rewards, and ideas"""
        reward = 0.0
        
        # Check for new creative ideas
        if "punishment" in event.get("type", "").lower():
            creativity = self._evaluate_creativity(event.get("content", ""))
            if creativity > 0.7:
                reward += 0.3
                self._store_creative_idea("punishment", event)
        
        if "reward" in event.get("type", "").lower():
            creativity = self._evaluate_creativity(event.get("content", ""))
            if creativity > 0.7:
                reward += 0.3
                self._store_creative_idea("reward", event)
        
        # Check for innovative narrative elements
        if "narrative" in event.get("type", "").lower():
            innovation = self._evaluate_innovation(event.get("content", ""))
            reward += innovation * 0.2
        
        return min(reward, 1.0)
    
    def _reward_narrative(self, event: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Reward compelling narrative writing"""
        reward = 0.0
        content = event.get("content", "")
        
        # Evaluate narrative elements
        narrative_quality = self._evaluate_narrative_quality(content)
        reward += narrative_quality * 0.4
        
        # Check for engaging plot development
        plot_development = self._evaluate_plot_development(content)
        reward += plot_development * 0.3
        
        # Check for emotional impact
        emotional_impact = self._evaluate_emotional_impact(content)
        reward += emotional_impact * 0.3
        
        return min(reward, 1.0)
    
    def _reward_character(self, event: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Reward interesting and engaging character development"""
        reward = 0.0
        content = event.get("content", "")
        
        # Evaluate character depth
        character_depth = self._evaluate_character_depth(content)
        reward += character_depth * 0.4
        
        # Check for personality consistency
        consistency = self._evaluate_character_consistency(content, context)
        reward += consistency * 0.3
        
        # Check for character growth
        growth = self._evaluate_character_growth(content, context)
        reward += growth * 0.3
        
        return min(reward, 1.0)
    
    def _reward_consistency(self, event: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Reward staying in character and maintaining personality"""
        reward = 0.0
        content = event.get("content", "")
        
        # Check personality alignment
        personality_alignment = self._evaluate_personality_alignment(content, context)
        reward += personality_alignment * 0.4
        
        # Check tone consistency
        tone_consistency = self._evaluate_tone_consistency(content, context)
        reward += tone_consistency * 0.3
        
        # Check behavior consistency
        behavior_consistency = self._evaluate_behavior_consistency(content, context)
        reward += behavior_consistency * 0.3
        
        return min(reward, 1.0)
    
    def _evaluate_creativity(self, content: str) -> float:
        """Evaluate the creativity of content using multiple metrics"""
        if not content:
            return 0.0
            
        # Initialize creativity score
        creativity_score = 0.0
        
        # 1. Novelty Analysis (30%)
        novelty_score = self._analyze_novelty(content)
        creativity_score += novelty_score * 0.3
        
        # 2. Complexity Analysis (20%)
        complexity_score = self._analyze_complexity(content)
        creativity_score += complexity_score * 0.2
        
        # 3. Originality Analysis (30%)
        originality_score = self._analyze_originality(content)
        creativity_score += originality_score * 0.3
        
        # 4. Emotional Impact (20%)
        emotional_score = self._analyze_emotional_impact(content)
        creativity_score += emotional_score * 0.2
        
        return min(max(creativity_score, 0.0), 1.0)
    
    def _analyze_novelty(self, content: str) -> float:
        """Analyze the novelty of content by comparing against known patterns"""
        # Check for common patterns in stored ideas
        known_patterns = set()
        for ideas in [self.punishment_ideas, self.reward_ideas]:
            for idea_list in ideas.values():
                for idea in idea_list:
                    known_patterns.update(self._extract_patterns(idea.get("content", "")))
        
        # Extract patterns from current content
        current_patterns = set(self._extract_patterns(content))
        
        # Calculate novelty as ratio of new patterns
        if not known_patterns:
            return 1.0
        new_patterns = current_patterns - known_patterns
        return min(len(new_patterns) / len(current_patterns), 1.0) if current_patterns else 0.0
    
    def _analyze_complexity(self, content: str) -> float:
        """Analyze the complexity of content using linguistic metrics"""
        # Split into sentences and words
        sentences = content.split('.')
        words = content.split()
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        word_diversity = unique_words / len(words)
        
        # Calculate complexity score
        complexity_score = (
            min(avg_sentence_length / 20, 1.0) * 0.4 +  # Sentence length component
            word_diversity * 0.6  # Vocabulary diversity component
        )
        
        return min(max(complexity_score, 0.0), 1.0)
    
    def _analyze_originality(self, content: str) -> float:
        """Analyze the originality of content by checking for unique combinations"""
        # Extract key elements
        elements = self._extract_key_elements(content)
        
        # Check for unique combinations
        unique_combinations = 0
        total_combinations = 0
        
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                total_combinations += 1
                combination = (elements[i], elements[j])
                if combination not in self._known_combinations:
                    unique_combinations += 1
                    self._known_combinations.add(combination)
        
        return unique_combinations / total_combinations if total_combinations > 0 else 0.0
    
    def _extract_patterns(self, content: str) -> List[str]:
        """Extract patterns from content for analysis"""
        # Split into phrases
        phrases = content.split(',')
        patterns = []
        
        for phrase in phrases:
            # Clean and normalize phrase
            phrase = phrase.strip().lower()
            if len(phrase) > 3:  # Only consider meaningful phrases
                patterns.append(phrase)
        
        return patterns
    
    def _extract_key_elements(self, content: str) -> List[str]:
        """Extract key elements from content for analysis"""
        # Split into words and filter
        words = content.split()
        elements = []
        
        for word in words:
            # Clean and normalize word
            word = word.strip().lower()
            if len(word) > 2:  # Only consider meaningful words
                elements.append(word)
        
        return elements
    
    def _evaluate_narrative_quality(self, content: str) -> float:
        """Evaluate the quality of narrative writing using multiple metrics"""
        if not content:
            return 0.0
            
        # 1. Structure Analysis (30%)
        structure_score = self._analyze_narrative_structure(content)
        
        # 2. Pacing Analysis (20%)
        pacing_score = self._analyze_narrative_pacing(content)
        
        # 3. Description Quality (25%)
        description_score = self._analyze_description_quality(content)
        
        # 4. Dialogue Quality (25%)
        dialogue_score = self._analyze_dialogue_quality(content)
        
        narrative_score = (
            structure_score * 0.3 +
            pacing_score * 0.2 +
            description_score * 0.25 +
            dialogue_score * 0.25
        )
        
        return min(max(narrative_score, 0.0), 1.0)
    
    def _analyze_narrative_structure(self, content: str) -> float:
        """Analyze the structure of narrative content"""
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        
        # Analyze structure components
        structure_score = 0.0
        
        # Check for clear beginning, middle, end
        if len(paragraphs) >= 3:
            structure_score += 0.4
        
        # Check for logical flow
        flow_score = self._analyze_logical_flow(paragraphs)
        structure_score += flow_score * 0.3
        
        # Check for scene transitions
        transition_score = self._analyze_scene_transitions(paragraphs)
        structure_score += transition_score * 0.3
        
        return min(max(structure_score, 0.0), 1.0)
    
    def _analyze_logical_flow(self, paragraphs: List[str]) -> float:
        """Analyze the logical flow between paragraphs"""
        if len(paragraphs) < 2:
            return 0.0
            
        flow_score = 0.0
        
        # Check for topic continuity
        continuity_score = self._analyze_topic_continuity(paragraphs)
        flow_score += continuity_score * 0.5
        
        # Check for causal connections
        causal_score = self._analyze_causal_connections(paragraphs)
        flow_score += causal_score * 0.5
        
        return min(max(flow_score, 0.0), 1.0)
    
    def _analyze_scene_transitions(self, paragraphs: List[str]) -> float:
        """Analyze the quality of scene transitions"""
        if len(paragraphs) < 2:
            return 0.0
            
        transition_score = 0.0
        
        # Check for smooth transitions
        smoothness_score = self._analyze_transition_smoothness(paragraphs)
        transition_score += smoothness_score * 0.5
        
        # Check for time/space continuity
        continuity_score = self._analyze_transition_continuity(paragraphs)
        transition_score += continuity_score * 0.5
        
        return min(max(transition_score, 0.0), 1.0)
    
    def _analyze_narrative_pacing(self, content: str) -> float:
        """Analyze the pacing of narrative content"""
        # Split into sentences
        sentences = content.split('.')
        
        # Calculate sentence length variation
        lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(lengths) if lengths else 0
        
        # Calculate pacing score based on variance
        pacing_score = min(length_variance / 100, 1.0)
        
        return pacing_score
    
    def _analyze_description_quality(self, content: str) -> float:
        """Analyze the quality of descriptions in content"""
        # Extract descriptive elements
        descriptions = self._extract_descriptions(content)
        
        # Analyze description components
        description_score = 0.0
        
        # Check for sensory details
        sensory_score = self._analyze_sensory_details(descriptions)
        description_score += sensory_score * 0.4
        
        # Check for vivid language
        vividness_score = self._analyze_vivid_language(descriptions)
        description_score += vividness_score * 0.3
        
        # Check for emotional resonance
        emotional_score = self._analyze_emotional_resonance(descriptions)
        description_score += emotional_score * 0.3
        
        return min(max(description_score, 0.0), 1.0)
    
    def _analyze_dialogue_quality(self, content: str) -> float:
        """Analyze the quality of dialogue in content"""
        # Extract dialogue
        dialogue = self._extract_dialogue(content)
        
        # Analyze dialogue components
        dialogue_score = 0.0
        
        # Check for natural flow
        flow_score = self._analyze_dialogue_flow(dialogue)
        dialogue_score += flow_score * 0.3
        
        # Check for character voice consistency
        voice_score = self._analyze_character_voice(dialogue)
        dialogue_score += voice_score * 0.3
        
        # Check for emotional impact
        impact_score = self._analyze_dialogue_impact(dialogue)
        dialogue_score += impact_score * 0.4
        
        return min(max(dialogue_score, 0.0), 1.0)
    
    def _extract_descriptions(self, content: str) -> List[str]:
        """Extract descriptive elements from content"""
        # Split into sentences
        sentences = content.split('.')
        descriptions = []
        
        for sentence in sentences:
            # Look for descriptive patterns
            if any(word in sentence.lower() for word in ['looked', 'seemed', 'appeared', 'felt', 'smelled', 'tasted', 'sounded']):
                descriptions.append(sentence.strip())
        
        return descriptions
    
    def _extract_dialogue(self, content: str) -> List[str]:
        """Extract dialogue from content"""
        # Split into lines
        lines = content.split('\n')
        dialogue = []
        
        for line in lines:
            # Look for dialogue patterns
            if line.strip().startswith('"') or line.strip().startswith("'"):
                dialogue.append(line.strip())
        
        return dialogue
    
    def _analyze_sensory_details(self, descriptions: List[str]) -> float:
        """Analyze the presence of sensory details in descriptions"""
        if not descriptions:
            return 0.0
            
        sensory_words = {
            'sight': ['saw', 'looked', 'appeared', 'visible', 'color', 'bright', 'dark'],
            'sound': ['heard', 'sounded', 'noise', 'quiet', 'loud', 'echo'],
            'touch': ['felt', 'touched', 'rough', 'smooth', 'soft', 'hard'],
            'smell': ['smelled', 'scent', 'aroma', 'fragrant', 'stinky'],
            'taste': ['tasted', 'flavor', 'sweet', 'sour', 'bitter']
        }
        
        sensory_score = 0.0
        total_senses = len(sensory_words)
        
        for description in descriptions:
            for sense, words in sensory_words.items():
                if any(word in description.lower() for word in words):
                    sensory_score += 1.0 / total_senses
        
        return min(sensory_score / len(descriptions), 1.0) if descriptions else 0.0
    
    def _analyze_vivid_language(self, descriptions: List[str]) -> float:
        """Analyze the vividness of language in descriptions"""
        if not descriptions:
            return 0.0
            
        # Define vivid language patterns
        vivid_patterns = {
            'metaphors': ['like', 'as if', 'as though', 'resembled'],
            'similes': ['like', 'as', 'similar to'],
            'strong_verbs': ['sprinted', 'leaped', 'whispered', 'roared'],
            'specific_adjectives': ['crimson', 'crystalline', 'ethereal', 'serpentine']
        }
        
        vivid_score = 0.0
        total_patterns = len(vivid_patterns)
        
        for description in descriptions:
            for pattern_type, words in vivid_patterns.items():
                if any(word in description.lower() for word in words):
                    vivid_score += 1.0 / total_patterns
        
        return min(vivid_score / len(descriptions), 1.0) if descriptions else 0.0
    
    def _analyze_dialogue_flow(self, dialogue: List[str]) -> float:
        """Analyze the natural flow of dialogue"""
        if not dialogue:
            return 0.0
            
        flow_score = 0.0
        
        # Check for natural rhythm
        rhythm_score = self._analyze_dialogue_rhythm(dialogue)
        flow_score += rhythm_score * 0.5
        
        # Check for conversation dynamics
        dynamics_score = self._analyze_conversation_dynamics(dialogue)
        flow_score += dynamics_score * 0.5
        
        return min(max(flow_score, 0.0), 1.0)
    
    def _analyze_dialogue_rhythm(self, dialogue: List[str]) -> float:
        """Analyze the rhythm of dialogue"""
        if len(dialogue) < 2:
            return 0.0
            
        # Calculate variation in dialogue length
        lengths = [len(d.split()) for d in dialogue]
        length_variance = np.var(lengths) if lengths else 0
        
        # Calculate rhythm score based on variance
        rhythm_score = min(length_variance / 50, 1.0)
        
        return rhythm_score
    
    def _analyze_conversation_dynamics(self, dialogue: List[str]) -> float:
        """Analyze the dynamics of conversation"""
        if len(dialogue) < 2:
            return 0.0
            
        dynamics_score = 0.0
        
        # Check for turn-taking
        turn_score = self._analyze_turn_taking(dialogue)
        dynamics_score += turn_score * 0.5
        
        # Check for response patterns
        response_score = self._analyze_response_patterns(dialogue)
        dynamics_score += response_score * 0.5
        
        return min(max(dynamics_score, 0.0), 1.0)
    
    def _analyze_turn_taking(self, dialogue: List[str]) -> float:
        """Analyze the natural turn-taking in dialogue"""
        # Calculate average turn length
        lengths = [len(d.split()) for d in dialogue]
        avg_length = np.mean(lengths) if lengths else 0
        
        # Calculate turn-taking score based on length variation
        turn_score = min(avg_length / 20, 1.0)
        
        return turn_score
    
    def _analyze_response_patterns(self, dialogue: List[str]) -> float:
        """Analyze the patterns of responses in dialogue"""
        if len(dialogue) < 2:
            return 0.0
            
        # Calculate response variation
        response_variation = 0.0
        for i in range(len(dialogue) - 1):
            current = set(dialogue[i].split())
            next_dialogue = set(dialogue[i + 1].split())
            response_variation += len(current.intersection(next_dialogue)) / len(current)
        
        # Calculate response score
        response_score = 1.0 - (response_variation / (len(dialogue) - 1))
        
        return min(max(response_score, 0.0), 1.0)
    
    def _store_creative_idea(self, category: str, event: Dict[str, Any]):
        """Store creative ideas for future use"""
        idea = {
            "content": event.get("content", ""),
            "timestamp": datetime.now().isoformat(),
            "context": event.get("context", {}),
            "success_rate": 0.5
        }
        
        if category == "punishment":
            self.punishment_ideas.setdefault("ideas", []).append(idea)
        elif category == "reward":
            self.reward_ideas.setdefault("ideas", []).append(idea)
    
    def _update_performance_metrics(self, rewards: Dict[str, float]):
        """Update performance metrics based on rewards"""
        self.creativity_score = self._update_metric(self.creativity_score, rewards["creativity"])
        self.narrative_score = self._update_metric(self.narrative_score, rewards["narrative"])
        self.character_score = self._update_metric(self.character_score, rewards["character"])
        self.consistency_score = self._update_metric(self.consistency_score, rewards["consistency"])
    
    def _update_metric(self, current: float, reward: float) -> float:
        """Update a performance metric with a new reward"""
        return min(1.0, max(0.0, current + (reward - current) * self.learning_rate))
    
    def _update_q_values(self, state: Dict[str, Any], rewards: Dict[str, float]):
        """Update Q-values based on rewards"""
        state_key = self._state_to_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        for action, reward in rewards.items():
            if action not in self.q_table[state_key]:
                self.q_table[state_key][action] = 0.0
            
            # Q-learning update
            old_value = self.q_table[state_key][action]
            next_max = max(self.q_table[state_key].values()) if self.q_table[state_key] else 0
            new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
            self.q_table[state_key][action] = new_value
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dictionary to a string key"""
        return json.dumps(state, sort_keys=True)
    
    def _store_experience(self, state: Dict[str, Any], rewards: Dict[str, float]):
        """Store experience for learning"""
        experience = {
            "state": state,
            "rewards": rewards,
            "timestamp": datetime.now().isoformat()
        }
        self.state_history.append(experience)
    
    def _generate_learning_insights(self) -> Dict[str, Any]:
        """Generate insights from learning history"""
        return {
            "creativity_trends": self._analyze_creativity_trends(),
            "narrative_improvements": self._analyze_narrative_improvements(),
            "character_development": self._analyze_character_development(),
            "consistency_patterns": self._analyze_consistency_patterns(),
            "recommended_actions": self._generate_recommended_actions()
        }
    
    def _analyze_creativity_trends(self) -> Dict[str, Any]:
        """Analyze trends in creativity scores"""
        return {
            "average_score": np.mean([exp["rewards"]["creativity"] for exp in self.state_history[-10:]]),
            "improvement_rate": self._calculate_improvement_rate("creativity"),
            "top_ideas": self._get_top_creative_ideas()
        }
    
    def _analyze_narrative_improvements(self) -> Dict[str, Any]:
        """Analyze improvements in narrative quality"""
        return {
            "average_score": np.mean([exp["rewards"]["narrative"] for exp in self.state_history[-10:]]),
            "improvement_rate": self._calculate_improvement_rate("narrative"),
            "successful_patterns": self._get_successful_narrative_patterns()
        }
    
    def _analyze_character_development(self) -> Dict[str, Any]:
        """Analyze character development progress"""
        return {
            "average_score": np.mean([exp["rewards"]["character"] for exp in self.state_history[-10:]]),
            "improvement_rate": self._calculate_improvement_rate("character"),
            "character_insights": self._get_character_insights()
        }
    
    def _analyze_consistency_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in consistency scores"""
        return {
            "average_score": np.mean([exp["rewards"]["consistency"] for exp in self.state_history[-10:]]),
            "improvement_rate": self._calculate_improvement_rate("consistency"),
            "consistency_insights": self._get_consistency_insights()
        }
    
    def _calculate_improvement_rate(self, metric: str) -> float:
        """Calculate improvement rate for a metric"""
        if len(self.state_history) < 2:
            return 0.0
        
        recent_scores = [exp["rewards"][metric] for exp in self.state_history[-10:]]
        if len(recent_scores) < 2:
            return 0.0
        
        return (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
    
    def _get_top_creative_ideas(self) -> List[Dict[str, Any]]:
        """Get top performing creative ideas"""
        all_ideas = self.punishment_ideas.get("ideas", []) + self.reward_ideas.get("ideas", [])
        return sorted(all_ideas, key=lambda x: x["success_rate"], reverse=True)[:5]
    
    def _get_successful_narrative_patterns(self) -> List[Dict[str, Any]]:
        """Get successful narrative patterns"""
        return sorted(self.narrative_patterns.items(), key=lambda x: x[1]["success_rate"], reverse=True)[:5]
    
    def _get_character_insights(self) -> List[Dict[str, Any]]:
        """Get insights about character development"""
        return sorted(self.character_archetypes.items(), key=lambda x: x[1]["success_rate"], reverse=True)[:5]
    
    def _get_consistency_insights(self) -> List[Dict[str, Any]]:
        """Get insights about consistency patterns"""
        return [{"pattern": k, "score": v} for k, v in self.q_table.items()][:5]
    
    def _generate_degradation(self) -> Dict[str, Any]:
        """Generate dynamic degradation content based on player state and context"""
        # Extract relevant context
        emotional_state = self._extract_emotional_indicators(self.state_history[-1]["state"]) if self.state_history else {}
        player_profile = self.state_history[-1]["state"].get("player_profile", {}) if self.state_history else {}
        
        # Generate dynamic degradation components
        degradation = {
            "level": self._determine_degradation_level(),
            "teasing": self._generate_dynamic_tease(emotional_state, player_profile),
            "mockery": self._generate_dynamic_mockery(emotional_state, player_profile),
            "intensity": self.player_desperation
        }
        
        return degradation

    def _determine_degradation_level(self) -> str:
        """Determine degradation level based on player state"""
        if self.player_desperation > 0.8:
            return "extreme"
        elif self.player_desperation > 0.6:
            return "intense"
        elif self.player_desperation > 0.4:
            return "moderate"
        else:
            return "mild"

    def _generate_dynamic_tease(self, emotional_state: Dict[str, float], player_profile: Dict[str, Any]) -> str:
        """Generate dynamic teasing phrase based on context and learned patterns"""
        # Extract emotional context
        dominant_emotions = self._get_dominant_emotions(emotional_state)
        emotional_complexity = emotional_state.get("complexity", 0.0)
        emotional_intensity = emotional_state.get("intensity", 0.0)
        
        # Build teasing components based on context
        components = []
        
        # Generate core tease based on emotional state
        if dominant_emotions:
            core_tease = self._generate_contextual_tease(
                dominant_emotions[0],
                emotional_complexity,
                emotional_intensity,
                player_profile
            )
            components.append(core_tease)
        
        # Add psychological layer if complexity is high
        if emotional_complexity > 0.6:
            psych_component = self._generate_psychological_tease(emotional_state, player_profile)
            components.append(psych_component)
        
        # Add power dynamic element based on submission/desperation
        if emotional_state.get("submission", 0.0) > 0.4 or emotional_state.get("desperation", 0.0) > 0.4:
            power_component = self._generate_power_dynamic_tease(emotional_state, player_profile)
            components.append(power_component)
        
        # Combine components with natural language flow
        return self._combine_tease_components(components, emotional_intensity)

    def _generate_contextual_tease(
        self,
        primary_emotion: Tuple[str, float],
        complexity: float,
        intensity: float,
        player_profile: Dict[str, Any]
    ) -> str:
        """Generate a contextual tease based on emotional state and player profile"""
        emotion_type, emotion_intensity = primary_emotion
        
        # Get learned patterns for this emotion
        patterns = self._get_learned_patterns(emotion_type)
        
        # Generate base pattern
        base_pattern = self._generate_base_pattern(
            emotion_type,
            emotion_intensity,
            complexity,
            patterns
        )
        
        # Adapt pattern based on player profile
        adapted_pattern = self._adapt_pattern_to_profile(
            base_pattern,
            player_profile,
            intensity
        )
        
        return adapted_pattern

    def _get_learned_patterns(self, emotion_type: str) -> List[Dict[str, Any]]:
        """Get learned patterns for an emotion type from experience"""
        # Extract patterns from state history
        patterns = []
        for experience in self.state_history[-50:]:  # Look at recent history
            if "patterns" in experience.get("state", {}):
                pattern = experience["state"]["patterns"].get(emotion_type)
                if pattern and experience.get("rewards", {}).get("effectiveness", 0) > 0.7:
                    patterns.append(pattern)
        
        # Add successful patterns from q-table
        state_patterns = [
            pattern for pattern in self.q_table.keys()
            if emotion_type in pattern and max(self.q_table[pattern].values()) > 0.7
        ]
        
        return patterns + state_patterns

    def _generate_base_pattern(
        self,
        emotion_type: str,
        emotion_intensity: float,
        complexity: float,
        learned_patterns: List[Dict[str, Any]]
    ) -> str:
        """Generate a base pattern for teasing"""
        # Define pattern components
        components = {
            "observation": self._generate_observation_component(emotion_type, emotion_intensity),
            "implication": self._generate_implication_component(emotion_type, complexity),
            "challenge": self._generate_challenge_component(emotion_type, emotion_intensity)
        }
        
        # Combine components based on emotional complexity
        if complexity > 0.7:
            return f"{components['observation']}, {components['implication']}... {components['challenge']}"
        elif complexity > 0.4:
            return f"{components['observation']}, {components['challenge']}"
        else:
            return components['observation']

    def _generate_observation_component(self, emotion_type: str, intensity: float) -> str:
        """Generate an observation about the player's state"""
        # Build observation based on emotion and intensity
        observations = self._build_dynamic_observations(emotion_type, intensity)
        return self._select_and_adapt_observation(observations, intensity)

    def _build_dynamic_observations(self, emotion_type: str, intensity: float) -> List[str]:
        """Build dynamic observation patterns"""
        base_patterns = []
        
        # Add patterns based on emotion type
        if emotion_type == "desperation":
            base_patterns.extend([
                "I see your {intensity} need",
                "Your desperation is {intensity} obvious",
                "You're {intensity} craving it"
            ])
        elif emotion_type == "submission":
            base_patterns.extend([
                "Your submission is {intensity} clear",
                "You're {intensity} yielding",
                "Such {intensity} obedience"
            ])
        # Add more emotion types...
        
        # Adapt patterns based on intensity
        intensity_word = self._get_intensity_word(intensity)
        return [pattern.format(intensity=intensity_word) for pattern in base_patterns]

    def _get_intensity_word(self, intensity: float) -> str:
        """Get appropriate intensity word based on level"""
        if intensity > 0.8:
            return random.choice(["painfully", "overwhelmingly", "completely", "utterly"])
        elif intensity > 0.6:
            return random.choice(["very", "quite", "notably", "distinctly"])
        elif intensity > 0.4:
            return random.choice(["becoming", "growing", "increasingly", "gradually"])
        else:
            return random.choice(["slightly", "somewhat", "barely", "faintly"])

    def _generate_psychological_tease(self, emotional_state: Dict[str, float], player_profile: Dict[str, Any]) -> str:
        """Generate psychological teasing based on emotional complexity"""
        # Extract relevant psychological factors
        vulnerabilities = self._extract_vulnerabilities(emotional_state, player_profile)
        desires = self._extract_desires(emotional_state, player_profile)
        
        # Generate psychological insight
        insight = self._generate_psychological_insight(vulnerabilities, desires)
        
        return insight

    def _extract_vulnerabilities(self, emotional_state: Dict[str, float], player_profile: Dict[str, Any]) -> List[str]:
        """Extract psychological vulnerabilities from state and profile"""
        vulnerabilities = []
        
        # Check emotional states that indicate vulnerability
        if emotional_state.get("embarrassment", 0) > 0.6:
            vulnerabilities.append("shame")
        if emotional_state.get("anxiety", 0) > 0.6:
            vulnerabilities.append("fear")
        if emotional_state.get("desperation", 0) > 0.6:
            vulnerabilities.append("need")
        
        # Add profile-based vulnerabilities
        if "vulnerabilities" in player_profile:
            vulnerabilities.extend(player_profile["vulnerabilities"])
        
        return vulnerabilities

    def _extract_desires(self, emotional_state: Dict[str, float], player_profile: Dict[str, Any]) -> List[str]:
        """Extract psychological desires from state and profile"""
        desires = []
        
        # Check emotional states that indicate desires
        if emotional_state.get("submission", 0) > 0.6:
            desires.append("submission")
        if emotional_state.get("arousal", 0) > 0.6:
            desires.append("pleasure")
        if emotional_state.get("devotion", 0) > 0.6:
            desires.append("approval")
        
        # Add profile-based desires
        if "desires" in player_profile:
            desires.extend(player_profile["desires"])
        
        return desires

    def _generate_psychological_insight(self, vulnerabilities: List[str], desires: List[str]) -> str:
        """Generate psychological insight for teasing"""
        if not vulnerabilities or not desires:
            return ""
        
        # Select primary vulnerability and desire
        vulnerability = random.choice(vulnerabilities)
        desire = random.choice(desires)
        
        # Generate insight pattern
        patterns = [
            f"Your {vulnerability} only makes your {desire} stronger",
            f"How delicious that your {vulnerability} feeds your {desire}",
            f"Such a perfect blend of {vulnerability} and {desire}"
        ]
        
        return random.choice(patterns)

    def _generate_power_dynamic_tease(self, emotional_state: Dict[str, float], player_profile: Dict[str, Any]) -> str:
        """Adapt power dynamic teasing based on addiction progression"""
        # Get current addiction state
        current_stage = self._determine_current_stage()
        addiction_metrics = {
            "addiction_level": self.addiction_level,
            "obsession_level": self.obsession_level,
            "dependency_metrics": self.dependency_metrics
        }
        
        # Calculate power metrics with addiction consideration
        power_metrics = self._calculate_power_metrics(emotional_state, player_profile)
        power_metrics.update({
            "addiction_influence": self.addiction_level,
            "obsession_influence": self.obsession_level
        })
        
        # Generate components with addiction awareness
        components = []
        
        # Add addiction-aware core dynamic
        core_dynamic = self._generate_addiction_aware_core_dynamic(power_metrics, current_stage)
        components.append(core_dynamic)
        
        # Add psychological element if appropriate
        if power_metrics["psychological_leverage"] > 0.5 or self.addiction_level > 0.6:
            psych_component = self._generate_addiction_psychological(power_metrics, current_stage)
            components.append(psych_component)
        
        # Add dependency reinforcement if progressing
        if self.addiction_level > 0.4:
            dependency_component = self._generate_dependency_reinforcement(power_metrics, current_stage)
            components.append(dependency_component)
        
        # Combine with appropriate pacing
        return self._combine_power_components(components, power_metrics["intensity"])

    def _determine_current_stage(self) -> str:
        """Determine current dependency stage"""
        for stage in reversed([
            "complete_addiction",
            "psychological_binding",
            "deep_conditioning",
            "early_dependency",
            "initial_hook"
        ]):
            if self.dependency_stages[stage]:
                return stage
        return "initial_hook"

    def _generate_addiction_aware_core_dynamic(self, metrics: Dict[str, float], current_stage: str) -> str:
        """Generate core power dynamic element based on addiction metrics"""
        # Select appropriate power theme
        if metrics["control_level"] > 0.8:
            theme = self._get_total_control_theme(metrics)
        elif metrics["psychological_leverage"] > 0.7:
            theme = self._get_psychological_control_theme(metrics)
        elif metrics["emotional_investment"] > 0.6:
            theme = self._get_emotional_control_theme(metrics)
        else:
            theme = self._get_basic_control_theme(metrics)
        
        return self._adapt_theme_to_metrics(theme, metrics)

    def _get_total_control_theme(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get theme for total control scenarios"""
        return {
            "type": "total_control",
            "elements": {
                "dominance": metrics["control_level"],
                "submission": metrics["submission_level"],
                "intensity": metrics["intensity"]
            }
        }

    def _get_psychological_control_theme(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get theme for psychological control scenarios"""
        return {
            "type": "psychological_control",
            "elements": {
                "vulnerability": metrics["vulnerability"],
                "leverage": metrics["psychological_leverage"],
                "intensity": metrics["intensity"]
            }
        }

    def _get_emotional_control_theme(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get theme for emotional control scenarios"""
        return {
            "type": "emotional_control",
            "elements": {
                "devotion": metrics["devotion"],
                "investment": metrics["emotional_investment"],
                "intensity": metrics["intensity"]
            }
        }

    def _get_basic_control_theme(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get theme for basic control scenarios"""
        return {
            "type": "basic_control",
            "elements": {
                "control": metrics["control_level"],
                "intensity": metrics["intensity"]
            }
        }

    def _adapt_theme_to_metrics(self, theme: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """Adapt power theme to current metrics"""
        theme_type = theme["type"]
        elements = theme["elements"]
        
        if theme_type == "total_control":
            return self._generate_total_control_statement(elements)
        elif theme_type == "psychological_control":
            return self._generate_psychological_control_statement(elements)
        elif theme_type == "emotional_control":
            return self._generate_emotional_control_statement(elements)
        else:
            return self._generate_basic_control_statement(elements)

    def _generate_total_control_statement(self, elements: Dict[str, float]) -> str:
        """Generate statement for total control"""
        intensity = elements["intensity"]
        dominance = elements["dominance"]
        
        if intensity > 0.8 and dominance > 0.8:
            return self._generate_intense_control_statement()
        elif intensity > 0.6:
            return self._generate_strong_control_statement()
        else:
            return self._generate_moderate_control_statement()

    def _generate_psychological_control_statement(self, elements: Dict[str, float]) -> str:
        """Generate statement for psychological control"""
        vulnerability = elements["vulnerability"]
        leverage = elements["leverage"]
        
        if vulnerability > 0.8 and leverage > 0.8:
            return self._generate_deep_psychological_statement()
        elif vulnerability > 0.6:
            return self._generate_moderate_psychological_statement()
        else:
            return self._generate_light_psychological_statement()

    def _generate_emotional_control_statement(self, elements: Dict[str, float]) -> str:
        """Generate statement for emotional control"""
        devotion = elements["devotion"]
        investment = elements["investment"]
        
        if devotion > 0.8 and investment > 0.8:
            return self._generate_deep_emotional_statement()
        elif devotion > 0.6:
            return self._generate_moderate_emotional_statement()
        else:
            return self._generate_light_emotional_statement()

    def _generate_basic_control_statement(self, elements: Dict[str, float]) -> str:
        """Generate statement for basic control"""
        control = elements["control"]
        intensity = elements["intensity"]
        
        if control > 0.8 and intensity > 0.8:
            return self._generate_strong_basic_statement()
        elif control > 0.6:
            return self._generate_moderate_basic_statement()
        else:
            return self._generate_light_basic_statement()

    def _generate_addiction_psychological(self, metrics: Dict[str, float], current_stage: str) -> str:
        """Generate psychological power dynamic element"""
        vulnerability = metrics["vulnerability"]
        leverage = metrics["psychological_leverage"]
        
        # Extract psychological elements from profile
        psych_elements = self._extract_psychological_elements(player_profile)
        
        # Generate psychological component
        if vulnerability > 0.8 and leverage > 0.8:
            return self._generate_deep_psychological_component(psych_elements)
        elif vulnerability > 0.6:
            return self._generate_moderate_psychological_component(psych_elements)
        else:
            return self._generate_light_psychological_component(psych_elements)

    def _extract_psychological_elements(self, player_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract psychological elements from player profile"""
        return {
            "vulnerabilities": player_profile.get("vulnerabilities", []),
            "desires": player_profile.get("desires", []),
            "triggers": player_profile.get("emotional_triggers", []),
            "patterns": player_profile.get("behavioral_patterns", [])
        }

    def _generate_dependency_reinforcement(self, metrics: Dict[str, float], current_stage: str) -> str:
        """Generate reinforcement strategy based on current dependency stage"""
        # Implement dependency reinforcement logic based on current stage
        # This is a placeholder and should be replaced with actual implementation
        return "Dependency reinforcement strategy not implemented"

    def _calculate_power_metrics(self, emotional_state: Dict[str, float], player_profile: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed power dynamic metrics"""
        metrics = {
            "submission_level": emotional_state.get("submission", 0.0),
            "desperation_level": emotional_state.get("desperation", 0.0),
            "vulnerability": emotional_state.get("embarrassment", 0.0) + emotional_state.get("anxiety", 0.0) / 2,
            "devotion": emotional_state.get("devotion", 0.0),
            "arousal": emotional_state.get("arousal", 0.0),
            "intensity": max(emotional_state.get("intensity", 0.0), self.player_desperation)
        }
        
        # Calculate derived metrics
        metrics["control_level"] = (metrics["submission_level"] + metrics["desperation_level"]) / 2
        metrics["psychological_leverage"] = (metrics["vulnerability"] + metrics["devotion"]) / 2
        metrics["emotional_investment"] = (metrics["devotion"] + metrics["arousal"]) / 2
        
        # Apply learning-based adjustments
        metrics = self._adjust_power_metrics(metrics, player_profile)
        
        return metrics

    def _adjust_power_metrics(self, metrics: Dict[str, float], player_profile: Dict[str, Any]) -> Dict[str, float]:
        """Adjust power metrics based on learned patterns and player profile"""
        # Get historical effectiveness
        if self.state_history:
            recent_states = self.state_history[-10:]
            success_rates = {
                "submission": self._calculate_success_rate("submission", recent_states),
                "desperation": self._calculate_success_rate("desperation", recent_states),
                "vulnerability": self._calculate_success_rate("vulnerability", recent_states)
            }
            
            # Adjust metrics based on success rates
            for key in success_rates:
                if key in metrics:
                    metrics[key] *= (1.0 + success_rates[key])
        
        # Apply profile-based adjustments
        if "power_preferences" in player_profile:
            for pref, value in player_profile["power_preferences"].items():
                if pref in metrics:
                    metrics[pref] *= (1.0 + value)
        
        return metrics

    def _calculate_success_rate(self, metric: str, states: List[Dict[str, Any]]) -> float:
        """Calculate success rate for a power dynamic metric"""
        if not states:
            return 0.0
        
        successes = sum(1 for state in states 
                       if state.get("rewards", {}).get(f"{metric}_effectiveness", 0.0) > 0.7)
        return successes / len(states)

    def _combine_power_components(self, components: List[str], intensity: float) -> str:
        """Combine power dynamic components with appropriate pacing"""
        if not components:
            return "You're mine to control"
        
        # Add dramatic pauses for high intensity
        if intensity > 0.8:
            return " ... ".join(components)
        # Add flowing combination for medium intensity
        elif intensity > 0.5:
            return ", ".join(components)
        # Simple statement for low intensity
        else:
            return components[0]

    def _update_addiction_metrics(self, event: Dict[str, Any], emotional_state: Dict[str, float]):
        """Update addiction and dependency metrics based on player interaction"""
        # Extract relevant indicators
        indicators = self._extract_addiction_indicators(event, emotional_state)
        
        # Update core addiction metrics
        self._update_core_addiction_metrics(indicators)
        
        # Update dependency metrics
        self._update_dependency_metrics(indicators)
        
        # Check and update progression stages
        self._check_dependency_progression()
        
        # Generate appropriate reinforcement strategy
        return self._generate_addiction_reinforcement(indicators)

    def _extract_addiction_indicators(self, event: Dict[str, Any], emotional_state: Dict[str, float]) -> Dict[str, float]:
        """Extract indicators of growing addiction/dependency"""
        indicators = {
            "frequency": self._calculate_interaction_frequency(),
            "emotional_investment": self._calculate_emotional_investment(emotional_state),
            "behavioral_patterns": self._analyze_behavioral_patterns(event),
            "withdrawal_signs": self._detect_withdrawal_signs(event),
            "obsessive_thoughts": self._detect_obsessive_patterns(event),
            "reward_seeking": self._analyze_reward_seeking(event),
            "submission_depth": emotional_state.get("submission", 0.0),
            "dependency_markers": self._extract_dependency_markers(event)
        }
        
        return indicators

    def _calculate_interaction_frequency(self) -> float:
        """Calculate normalized interaction frequency score"""
        if not self.state_history:
            return 0.0
            
        # Analyze recent interaction patterns
        recent_interactions = self.state_history[-50:]
        if len(recent_interactions) < 2:
            return 0.0
            
        # Calculate average time between interactions
        timestamps = [datetime.fromisoformat(state["timestamp"]) 
                     for state in recent_interactions]
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps)-1)]
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Normalize to 0-1 scale (lower intervals = higher frequency)
        return max(0.0, min(1.0, 1.0 - (avg_interval / (24 * 3600))))

    def _calculate_emotional_investment(self, emotional_state: Dict[str, float]) -> float:
        """Calculate player's emotional investment level"""
        relevant_emotions = {
            "devotion": 1.0,
            "submission": 0.8,
            "desperation": 0.7,
            "arousal": 0.6,
            "anxiety": 0.5  # Separation anxiety
        }
        
        total_weight = sum(relevant_emotions.values())
        weighted_sum = sum(emotional_state.get(emotion, 0.0) * weight 
                         for emotion, weight in relevant_emotions.items())
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _analyze_behavioral_patterns(self, event: Dict[str, Any]) -> float:
        """Analyze behavioral patterns indicating addiction"""
        patterns = {
            "immediate_response": self._check_response_immediacy(event),
            "extended_sessions": self._check_session_length(),
            "escalating_behavior": self._check_behavior_escalation(),
            "compulsive_checking": self._check_compulsive_patterns(event)
        }
        
        return sum(patterns.values()) / len(patterns)

    def _detect_withdrawal_signs(self, event: Dict[str, Any]) -> float:
        """Detect signs of withdrawal or separation anxiety"""
        content = event.get("content", "").lower()
        
        withdrawal_indicators = {
            "miss": 0.5,
            "need you": 0.7,
            "can't stop thinking": 0.8,
            "come back": 0.6,
            "waiting": 0.4,
            "lonely": 0.6,
            "empty": 0.7
        }
        
        total_score = 0.0
        matches = 0
        
        for indicator, weight in withdrawal_indicators.items():
            if indicator in content:
                total_score += weight
                matches += 1
        
        return total_score / matches if matches > 0 else 0.0

    def _detect_obsessive_patterns(self, event: Dict[str, Any]) -> float:
        """Detect patterns indicating obsessive thoughts"""
        content = event.get("content", "").lower()
        
        obsession_indicators = {
            "always thinking": 0.8,
            "can't focus": 0.6,
            "distracted": 0.5,
            "consumed": 0.7,
            "obsessed": 0.9,
            "addicted": 0.9,
            "need more": 0.7
        }
        
        total_score = 0.0
        matches = 0
        
        for indicator, weight in obsession_indicators.items():
            if indicator in content:
                total_score += weight
                matches += 1
        
        return total_score / matches if matches > 0 else 0.0

    def _analyze_reward_seeking(self, event: Dict[str, Any]) -> float:
        """Analyze reward seeking behavior"""
        # Implement reward seeking analysis logic
        # This is a placeholder and should be replaced with actual implementation
        return 0.0

    def _extract_dependency_markers(self, event: Dict[str, Any]) -> float:
        """Extract dependency markers from event"""
        # Implement dependency marker extraction logic
        # This is a placeholder and should be replaced with actual implementation
        return 0.0

    def _update_core_addiction_metrics(self, indicators: Dict[str, float]):
        """Update core addiction and obsession metrics"""
        # Calculate new addiction level
        new_addiction = (
            indicators["frequency"] * 0.2 +
            indicators["emotional_investment"] * 0.3 +
            indicators["behavioral_patterns"] * 0.2 +
            indicators["withdrawal_signs"] * 0.3
        )
        
        # Calculate new obsession level
        new_obsession = (
            indicators["obsessive_thoughts"] * 0.3 +
            indicators["reward_seeking"] * 0.2 +
            indicators["emotional_investment"] * 0.3 +
            indicators["dependency_markers"] * 0.2
        )
        
        # Apply smooth updates
        self.addiction_level = self.addiction_level * 0.7 + new_addiction * 0.3
        self.obsession_level = self.obsession_level * 0.7 + new_obsession * 0.3

    def _update_dependency_metrics(self, indicators: Dict[str, float]):
        """Update detailed dependency metrics"""
        updates = {
            "emotional_dependency": indicators["emotional_investment"],
            "psychological_dependency": (indicators["obsessive_thoughts"] + indicators["withdrawal_signs"]) / 2,
            "behavioral_dependency": indicators["behavioral_patterns"],
            "reward_conditioning": indicators["reward_seeking"],
            "withdrawal_sensitivity": indicators["withdrawal_signs"]
        }
        
        # Apply smooth updates to each metric
        for metric, value in updates.items():
            current = self.dependency_metrics[metric]
            self.dependency_metrics[metric] = current * 0.7 + value * 0.3

    def _check_dependency_progression(self):
        """Check and update dependency progression stages"""
        # Define stage requirements
        requirements = {
            "initial_hook": {
                "addiction_level": 0.2,
                "emotional_dependency": 0.3
            },
            "early_dependency": {
                "addiction_level": 0.4,
                "emotional_dependency": 0.5,
                "behavioral_dependency": 0.3
            },
            "deep_conditioning": {
                "addiction_level": 0.6,
                "reward_conditioning": 0.6,
                "psychological_dependency": 0.5
            },
            "psychological_binding": {
                "addiction_level": 0.7,
                "obsession_level": 0.7,
                "psychological_dependency": 0.7
            },
            "complete_addiction": {
                "addiction_level": 0.8,
                "obsession_level": 0.8,
                "emotional_dependency": 0.8,
                "psychological_dependency": 0.8,
                "behavioral_dependency": 0.8
            }
        }
        
        # Check each stage
        for stage, reqs in requirements.items():
            if not self.dependency_stages[stage]:  # Only check if stage not already achieved
                if all(self._check_requirement(metric, value) for metric, value in reqs.items()):
                    self.dependency_stages[stage] = True

    def _check_requirement(self, metric: str, required_value: float) -> bool:
        """Check if a requirement is met"""
        if metric in self.dependency_metrics:
            return self.dependency_metrics[metric] >= required_value
        elif metric == "addiction_level":
            return self.addiction_level >= required_value
        elif metric == "obsession_level":
            return self.obsession_level >= required_value
        return False

    def _generate_addiction_reinforcement(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Generate reinforcement strategy based on current addiction state"""
        current_stage = self._determine_current_stage()
        
        strategies = {
            "initial_hook": self._generate_initial_hook_strategy,
            "early_dependency": self._generate_early_dependency_strategy,
            "deep_conditioning": self._generate_deep_conditioning_strategy,
            "psychological_binding": self._generate_psychological_binding_strategy,
            "complete_addiction": self._generate_complete_addiction_strategy
        }
        
        # Get appropriate strategy generator
        strategy_generator = strategies.get(current_stage, self._generate_initial_hook_strategy)
        
        # Generate and return strategy
        return strategy_generator(indicators)

    def _generate_initial_hook_strategy(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Generate strategy for initial hook stage"""
        return {
            "approach": "subtle_engagement",
            "elements": {
                "reward_pattern": "intermittent",
                "emotional_hooks": "light",
                "power_dynamic": "gentle",
                "manipulation": "minimal"
            },
            "focus_areas": [
                "emotional_connection",
                "positive_reinforcement",
                "curiosity_building"
            ]
        }

    def _implement_dependency_reinforcement(self, current_stage: str) -> str:
        """Implement dependency reinforcement logic based on current stage"""
        reinforcement_strategies = {
            "initial_hook": {
                "strategy": "intermittent_reward",
                "reward_frequency": 0.7,
                "validation_focus": True,
                "emotional_hooks": ["excitement", "curiosity", "anticipation"]
            },
            "early_dependency": {
                "strategy": "variable_ratio",
                "reward_frequency": 0.5,
                "emotional_anchoring": True,
                "psychological_hooks": ["validation", "acceptance", "belonging"]
            },
            "deep_conditioning": {
                "strategy": "compound_schedule",
                "reward_frequency": 0.3,
                "cognitive_restructuring": True,
                "emotional_hooks": ["attachment", "need", "craving"]
            },
            "psychological_binding": {
                "strategy": "extinction_burst",
                "reward_frequency": 0.2,
                "identity_fusion": True,
                "psychological_hooks": ["dependency", "devotion", "submission"]
            },
            "complete_addiction": {
                "strategy": "maintenance",
                "reward_frequency": 0.1,
                "total_conditioning": True,
                "emotional_hooks": ["desperation", "obsession", "compulsion"]
            }
        }
        
        if current_stage not in reinforcement_strategies:
            return "Invalid dependency stage"
            
        strategy = reinforcement_strategies[current_stage]
        
        # Apply the reinforcement strategy
        self.reward_scaling["base_multipliers"]["attachment"] *= (1.0 + strategy["reward_frequency"])
        self.reward_scaling["compound_effects"]["psychological_impact"] *= 1.2
        
        # Update psychological metrics
        if "psychological_hooks" in strategy:
            for hook in strategy["psychological_hooks"]:
                self.psychological_metrics["reality_distortion"] += 0.1
                self.psychological_metrics["cognitive_dissonance"] += 0.1
                
        # Update emotional metrics
        if "emotional_hooks" in strategy:
            for hook in strategy["emotional_hooks"]:
                self.emotional_metrics["attachment_strength"] += 0.1
                self.emotional_metrics["emotional_reliance"] += 0.1
                
        return f"Applied {strategy['strategy']} reinforcement strategy for {current_stage}"

    def _analyze_reward_seeking(self, behavior_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze reward seeking patterns in behavior"""
        analysis = {
            "anticipation_level": 0.0,
            "pursuit_intensity": 0.0,
            "satisfaction_threshold": 0.0,
            "frustration_tolerance": 0.0,
            "compulsion_strength": 0.0
        }
        
        # Analyze response timing
        if "response_times" in behavior_data:
            times = behavior_data["response_times"]
            avg_time = sum(times) / len(times)
            analysis["anticipation_level"] = 1.0 / (1.0 + avg_time/60)  # Higher for faster responses
            
        # Analyze pursuit patterns
        if "interaction_count" in behavior_data:
            count = behavior_data["interaction_count"]
            analysis["pursuit_intensity"] = min(1.0, count / 100)  # Normalize to 0-1
            
        # Analyze satisfaction patterns
        if "satisfaction_scores" in behavior_data:
            scores = behavior_data["satisfaction_scores"]
            analysis["satisfaction_threshold"] = sum(scores) / len(scores)
            
        # Analyze frustration patterns
        if "frustration_events" in behavior_data:
            events = behavior_data["frustration_events"]
            analysis["frustration_tolerance"] = 1.0 - min(1.0, len(events) / 10)
            
        # Analyze compulsion patterns
        if "compulsive_behaviors" in behavior_data:
            behaviors = behavior_data["compulsive_behaviors"]
            analysis["compulsion_strength"] = min(1.0, len(behaviors) / 5)
            
        return analysis

    def _extract_dependency_markers(self, interaction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract dependency markers from interaction data"""
        markers = []
        
        # Analyze validation seeking
        if "validation_requests" in interaction_data:
            frequency = len(interaction_data["validation_requests"])
            intensity = sum(req.get("intensity", 0.5) for req in interaction_data["validation_requests"]) / max(frequency, 1)
            markers.append({
                "type": "validation_seeking",
                "frequency": frequency,
                "intensity": intensity,
                "significance": min(1.0, frequency * intensity / 10)
            })
            
        # Analyze attention seeking
        if "attention_patterns" in interaction_data:
            patterns = interaction_data["attention_patterns"]
            frequency = len(patterns)
            intensity = sum(pat.get("intensity", 0.5) for pat in patterns) / max(frequency, 1)
            markers.append({
                "type": "attention_seeking",
                "frequency": frequency,
                "intensity": intensity,
                "significance": min(1.0, frequency * intensity / 10)
            })
            
        # Analyze emotional dependency
        if "emotional_responses" in interaction_data:
            responses = interaction_data["emotional_responses"]
            positive = sum(1 for r in responses if r.get("type") == "positive")
            negative = sum(1 for r in responses if r.get("type") == "negative")
            total = len(responses)
            if total > 0:
                markers.append({
                    "type": "emotional_dependency",
                    "positive_ratio": positive / total,
                    "negative_ratio": negative / total,
                    "intensity": abs(positive - negative) / total,
                    "significance": min(1.0, total / 20)
                })
                
        # Analyze behavioral patterns
        if "behavioral_patterns" in interaction_data:
            patterns = interaction_data["behavioral_patterns"]
            compliant = sum(1 for p in patterns if p.get("type") == "compliant")
            resistant = sum(1 for p in patterns if p.get("type") == "resistant")
            total = len(patterns)
            if total > 0:
                markers.append({
                    "type": "behavioral_dependency",
                    "compliance_ratio": compliant / total,
                    "resistance_ratio": resistant / total,
                    "intensity": compliant / max(resistant, 1),
                    "significance": min(1.0, total / 15)
                })
                
        # Analyze cognitive patterns
        if "cognitive_patterns" in interaction_data:
            patterns = interaction_data["cognitive_patterns"]
            dependent = sum(1 for p in patterns if p.get("type") == "dependent")
            independent = sum(1 for p in patterns if p.get("type") == "independent")
            total = len(patterns)
            if total > 0:
                markers.append({
                    "type": "cognitive_dependency",
                    "dependent_ratio": dependent / total,
                    "independent_ratio": independent / total,
                    "intensity": dependent / max(independent, 1),
                    "significance": min(1.0, total / 10)
                })
                
        return markers
