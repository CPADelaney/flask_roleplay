# logic/dynamic_relationships.py
"""
Dynamic Relationships System - Optimized Version
Addresses performance, scaling, and token cost issues.
"""

import json
import logging
import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from collections import deque
import math
import statistics
import yaml
from pathlib import Path

import asyncpg

# Import core systems
from lore.core import canon
from lore.core.lore_system import LoreSystem
from logic.stats_logic import calculate_social_insight, apply_stat_changes
from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem

# Agents SDK imports
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper,
    trace
)

logger = logging.getLogger(__name__)

# ========================================================================
# CONFIGURATION SYSTEM
# ========================================================================

class RelationshipConfig:
    """Centralized configuration for tunable parameters"""
    
    _instance = None
    _config = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_config()
        return cls._instance
    
    def load_config(self, config_path: str = "config/relationships.yaml"):
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        # Default configuration
        self._config = {
            "thresholds": {
                "significance": 10,  # Minimum change to trigger events
                "momentum_trigger": 15,  # Momentum magnitude for turning points
                "decay_start_hours": 24,  # Hours before decay starts
                "cache_ttl_seconds": 300,
            },
            "decay_rates": {
                "frequency_daily": 0.05,  # 5% per day, not per tick
                "intimacy_daily": 0.02,   # 2% per day
                "fascination_daily": 0.01, # 1% per day
            },
            "velocity_factors": {
                "scale": 0.01,  # Scale factor for velocity changes
                "decay": 0.9,   # Velocity decay per update
                "max": 5.0,     # Maximum velocity magnitude
            },
            "pattern_thresholds": {
                "push_pull_alternations": 0.4,
                "slow_burn_consistency": 5,
                "explosive_chemistry_affection": 70,
                "explosive_chemistry_volatility": 60,
                "frenemies_conflict": 50,
                "frenemies_respect": 60,
                "rollercoaster_variance": 400,
            },
            "archetype_requirements": {
                # Moved from hardcoded values
                "soulmates": {
                    "trust": [80, None],
                    "affection": [80, None],
                    "intimacy": [85, None]
                },
                # ... etc
            }
        }
        
        # Load from file if exists
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    self._merge_config(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
    
    def _merge_config(self, loaded: Dict):
        """Merge loaded config with defaults"""
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        deep_merge(self._config, loaded)
    
    def get(self, path: str, default=None):
        """Get config value by dot notation path"""
        parts = path.split('.')
        value = self._config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

# Global config instance
config = RelationshipConfig.get_instance()

# ========================================================================
# OPTIMIZED DATA STRUCTURES
# ========================================================================

@dataclass
class RelationshipDimensions:
    """Multi-dimensional representation of a relationship state"""
    # Emotional Dimensions (-100 to +100)
    trust: float = 0.0
    respect: float = 0.0
    affection: float = 0.0
    fascination: float = 0.0
    
    # Power Dynamics (-100 to +100)
    influence: float = 0.0      # -100 = they control you, +100 = you control them
    
    # Mutual Metrics (0 to 100)
    dependence: float = 0.0
    intimacy: float = 0.0
    frequency: float = 0.0
    volatility: float = 0.0
    
    # Hidden Tensions (0 to 100)
    unresolved_conflict: float = 0.0
    hidden_agendas: float = 0.0
    
    def clamp(self):
        """Ensure all values are within valid ranges"""
        # Bidirectional dimensions
        for attr in ['trust', 'respect', 'affection', 'fascination', 'influence']:
            val = getattr(self, attr)
            setattr(self, attr, max(-100.0, min(100.0, val)))
        
        # Unidirectional dimensions
        for attr in ['dependence', 'intimacy', 'frequency', 'volatility', 
                     'unresolved_conflict', 'hidden_agendas']:
            val = getattr(self, attr)
            setattr(self, attr, max(0.0, min(100.0, val)))
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v != 0.0}  # Only non-zero values
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'RelationshipDimensions':
        """Create from dictionary"""
        dims = cls()
        for key, value in data.items():
            if hasattr(dims, key):
                setattr(dims, key, value)
        return dims
    
    def diff(self, other: 'RelationshipDimensions') -> Dict[str, float]:
        """Calculate difference between two dimension states"""
        diff = {}
        for attr in ['trust', 'respect', 'affection', 'fascination', 'influence',
                    'dependence', 'intimacy', 'frequency', 'volatility',
                    'unresolved_conflict', 'hidden_agendas']:
            old_val = getattr(other, attr)
            new_val = getattr(self, attr)
            if abs(new_val - old_val) > 0.1:  # Ignore tiny changes
                diff[attr] = new_val - old_val
        return diff

@dataclass
class RelationshipMomentum:
    """Tracks the velocity and direction of relationship changes"""
    # Store only active velocities to save space
    velocities: Dict[str, float] = field(default_factory=dict)
    inertia: float = 50.0
    
    def get_velocity(self, dimension: str) -> float:
        """Get velocity for a dimension"""
        return self.velocities.get(dimension, 0.0)
    
    def update_velocity(self, dimension: str, change: float):
        """Update velocity with proper scaling and capping"""
        decay = config.get('velocity_factors.decay', 0.9)
        scale = config.get('velocity_factors.scale', 0.01)
        max_velocity = config.get('velocity_factors.max', 5.0)
        
        current = self.velocities.get(dimension, 0.0)
        new_velocity = (current * decay) + (change * scale)
        
        # Cap velocity
        new_velocity = max(-max_velocity, min(max_velocity, new_velocity))
        
        # Remove if negligible
        if abs(new_velocity) < 0.01:
            self.velocities.pop(dimension, None)
        else:
            self.velocities[dimension] = new_velocity
    
    def apply_momentum(self, dimensions: RelationshipDimensions, 
                      time_delta: float = 1.0) -> RelationshipDimensions:
        """Apply natural drift based on momentum"""
        drift_factor = (1 - self.inertia / 100) * time_delta
        
        # Apply velocity-based drift
        for dimension, velocity in self.velocities.items():
            if hasattr(dimensions, dimension):
                current = getattr(dimensions, dimension)
                setattr(dimensions, dimension, current + velocity * drift_factor)
        
        return dimensions
    
    def get_magnitude(self) -> float:
        """Get overall momentum magnitude"""
        if not self.velocities:
            return 0.0
        return math.sqrt(sum(v**2 for v in self.velocities.values()))

@dataclass
class CompactRelationshipContext:
    """Optimized context storage - only store deltas from base"""
    base_dimensions: RelationshipDimensions = field(default_factory=RelationshipDimensions)
    context_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_context(self, situation: str) -> RelationshipDimensions:
        """Get dimensions for a specific context"""
        if situation not in self.context_deltas:
            return self.base_dimensions
        
        # Apply deltas to base
        dims = RelationshipDimensions(**asdict(self.base_dimensions))
        for attr, delta in self.context_deltas[situation].items():
            if hasattr(dims, attr):
                setattr(dims, attr, getattr(dims, attr) + delta)
        
        dims.clamp()
        return dims
    
    def set_context_delta(self, situation: str, dimension: str, delta: float):
        """Set a context-specific delta"""
        if situation not in self.context_deltas:
            self.context_deltas[situation] = {}
        
        if abs(delta) < 0.1:  # Ignore tiny deltas
            self.context_deltas[situation].pop(dimension, None)
        else:
            self.context_deltas[situation][dimension] = delta
        
        # Clean up empty contexts
        if not self.context_deltas[situation]:
            self.context_deltas.pop(situation, None)

@dataclass
class CompactRelationshipHistory:
    """Optimized history storage"""
    # Only store recent snapshots with significant changes
    significant_snapshots: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_interactions: deque = field(default_factory=lambda: deque(maxlen=50))
    active_patterns: Set[str] = field(default_factory=set)
    pattern_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def add_snapshot_if_significant(self, dimensions: RelationshipDimensions, 
                                   previous: Optional[RelationshipDimensions] = None):
        """Only store snapshots with significant changes"""
        if previous is None and self.significant_snapshots:
            # Compare with last snapshot
            last = RelationshipDimensions.from_dict(
                self.significant_snapshots[-1]['dimensions']
            )
            diff = dimensions.diff(last)
        elif previous:
            diff = dimensions.diff(previous)
        else:
            diff = {'initial': True}
        
        # Check if significant
        significance_threshold = config.get('thresholds.significance', 10)
        is_significant = any(abs(v) >= significance_threshold for v in diff.values() 
                           if isinstance(v, (int, float)))
        
        if is_significant or 'initial' in diff:
            self.significant_snapshots.append({
                'timestamp': datetime.now(),
                'dimensions': dimensions.to_dict(),
                'diff': diff
            })
    
    def add_interaction(self, interaction_type: str, context: str = "casual"):
        """Store interaction efficiently"""
        self.recent_interactions.append({
            'timestamp': datetime.now(),
            'type': interaction_type,
            'context': context
        })

@dataclass
class RelationshipState:
    """Optimized relationship state"""
    # Entity identifiers
    entity1_type: str
    entity1_id: int
    entity2_type: str
    entity2_id: int
    
    # Canonical ordering to prevent duplicates
    @property
    def canonical_key(self) -> str:
        """Get canonical key for this relationship"""
        # Always order by (type, id) tuple
        e1 = (self.entity1_type, self.entity1_id)
        e2 = (self.entity2_type, self.entity2_id)
        
        if e1 <= e2:
            return f"{e1[0]}_{e1[1]}_{e2[0]}_{e2[1]}"
        else:
            return f"{e2[0]}_{e2[1]}_{e1[0]}_{e1[1]}"
    
    # Core state
    dimensions: RelationshipDimensions = field(default_factory=RelationshipDimensions)
    momentum: RelationshipMomentum = field(default_factory=RelationshipMomentum)
    contexts: CompactRelationshipContext = field(default_factory=CompactRelationshipContext)
    history: CompactRelationshipHistory = field(default_factory=CompactRelationshipHistory)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    version: int = 0  # For cache invalidation
    
    # Computed properties stored separately
    active_archetypes: Set[str] = field(default_factory=set)
    
    def get_duration_days(self) -> int:
        """How long this relationship has existed"""
        return (datetime.now() - self.created_at).days
    
    def to_summary(self) -> Dict[str, Any]:
        """Get lightweight summary for agent tools"""
        return {
            'dimensions': self.dimensions.to_dict(),
            'momentum_magnitude': self.momentum.get_magnitude(),
            'patterns': list(self.history.active_patterns),
            'archetypes': list(self.active_archetypes),
            'duration_days': self.get_duration_days()
        }

# ========================================================================
# PATTERN DETECTION (OPTIMIZED)
# ========================================================================

class RelationshipPatternDetector:
    """Detects emergent patterns with configurable thresholds"""
    
    @staticmethod
    def detect_patterns(history: CompactRelationshipHistory) -> Set[str]:
        """Analyze history to detect active patterns"""
        patterns = set()
        
        # Need at least a few snapshots
        if len(history.significant_snapshots) < 3:
            return patterns
        
        snapshots = list(history.significant_snapshots)
        
        # Check each pattern
        for pattern_name, detector_func in [
            ("push_pull", RelationshipPatternDetector._detect_push_pull),
            ("slow_burn", RelationshipPatternDetector._detect_slow_burn),
            ("explosive_chemistry", RelationshipPatternDetector._detect_explosive_chemistry),
            ("frenemies", RelationshipPatternDetector._detect_frenemies),
            ("emotional_rollercoaster", RelationshipPatternDetector._detect_emotional_rollercoaster),
            ("growing_distance", RelationshipPatternDetector._detect_growing_distance),
        ]:
            if detector_func(snapshots):
                patterns.add(pattern_name)
        
        return patterns
    
    @staticmethod
    def _detect_push_pull(snapshots: List[Dict]) -> bool:
        """One gets close, the other pulls away"""
        if len(snapshots) < 5:
            return False
        
        intimacy_changes = []
        for i in range(1, len(snapshots)):
            prev = snapshots[i-1]['dimensions'].get('intimacy', 0)
            curr = snapshots[i]['dimensions'].get('intimacy', 0)
            if 'diff' in snapshots[i] and 'intimacy' in snapshots[i]['diff']:
                intimacy_changes.append(snapshots[i]['diff']['intimacy'])
        
        if len(intimacy_changes) < 4:
            return False
        
        # Look for alternating signs
        alternations = 0
        for i in range(1, len(intimacy_changes)):
            if intimacy_changes[i] * intimacy_changes[i-1] < 0:  # Different signs
                alternations += 1
        
        threshold = config.get('pattern_thresholds.push_pull_alternations', 0.4)
        return alternations >= len(intimacy_changes) * threshold
    
    @staticmethod
    def _detect_slow_burn(snapshots: List[Dict]) -> bool:
        """Gradual consistent improvement"""
        if len(snapshots) < config.get('pattern_thresholds.slow_burn_consistency', 5):
            return False
        
        # Check trust trajectory
        trust_values = [s['dimensions'].get('trust', 0) for s in snapshots[-5:]]
        
        # All increasing (allowing small decreases)
        increasing = all(trust_values[i] >= trust_values[i-1] - 5 
                        for i in range(1, len(trust_values)))
        
        # But not too fast
        total_change = trust_values[-1] - trust_values[0]
        gradual = total_change < 30  # Less than 30 points over 5 snapshots
        
        return increasing and gradual
    
    @staticmethod
    def _detect_explosive_chemistry(snapshots: List[Dict]) -> bool:
        """High intensity with high volatility"""
        if len(snapshots) < 3:
            return False
        
        recent = snapshots[-3:]
        avg_affection = statistics.mean(s['dimensions'].get('affection', 0) for s in recent)
        avg_volatility = statistics.mean(s['dimensions'].get('volatility', 0) for s in recent)
        
        affection_threshold = config.get('pattern_thresholds.explosive_chemistry_affection', 70)
        volatility_threshold = config.get('pattern_thresholds.explosive_chemistry_volatility', 60)
        
        return avg_affection > affection_threshold and avg_volatility > volatility_threshold
    
    @staticmethod
    def _detect_frenemies(snapshots: List[Dict]) -> bool:
        """High conflict but also high respect"""
        if len(snapshots) < 3:
            return False
        
        recent = snapshots[-3:]
        avg_conflict = statistics.mean(s['dimensions'].get('unresolved_conflict', 0) for s in recent)
        avg_respect = statistics.mean(s['dimensions'].get('respect', 0) for s in recent)
        
        conflict_threshold = config.get('pattern_thresholds.frenemies_conflict', 50)
        respect_threshold = config.get('pattern_thresholds.frenemies_respect', 60)
        
        return avg_conflict > conflict_threshold and avg_respect > respect_threshold
    
    @staticmethod
    def _detect_emotional_rollercoaster(snapshots: List[Dict]) -> bool:
        """Wild swings in emotional dimensions"""
        if len(snapshots) < 5:
            return False
        
        affection_values = [s['dimensions'].get('affection', 0) for s in snapshots[-5:]]
        
        # Calculate variance using statistics module
        if len(set(affection_values)) == 1:  # All same value
            return False
        
        variance = statistics.variance(affection_values)
        threshold = config.get('pattern_thresholds.rollercoaster_variance', 400)
        
        return variance > threshold
    
    @staticmethod
    def _detect_growing_distance(snapshots: List[Dict]) -> bool:
        """Declining frequency and intimacy"""
        if len(snapshots) < 4:
            return False
        
        recent = snapshots[-4:]
        
        # Check both frequency and intimacy are declining
        freq_declining = all(
            recent[i]['dimensions'].get('frequency', 0) <= 
            recent[i-1]['dimensions'].get('frequency', 0) + 2
            for i in range(1, len(recent))
        )
        
        intimacy_declining = all(
            recent[i]['dimensions'].get('intimacy', 0) <= 
            recent[i-1]['dimensions'].get('intimacy', 0) + 2
            for i in range(1, len(recent))
        )
        
        return freq_declining and intimacy_declining

# ========================================================================
# ARCHETYPE SYSTEM (WITH CONFIG)
# ========================================================================

class RelationshipArchetypes:
    """Different ways relationships can be meaningful"""
    
    @classmethod
    def get_archetypes(cls) -> Dict[str, Dict[str, Any]]:
        """Get archetype definitions from config or defaults"""
        return config.get('archetype_requirements', cls._get_default_archetypes())
    
    @classmethod
    def _get_default_archetypes(cls) -> Dict[str, Dict[str, Any]]:
        """Default archetype definitions"""
        return {
            "soulmates": {
                "requirements": {
                    "trust": [80, None],
                    "affection": [80, None],
                    "intimacy": [85, None]
                },
                "description": "Deep understanding and unconditional acceptance",
                "rewards": {
                    "abilities": ["Emotional Resonance", "Unspoken Understanding"],
                    "stat_bonuses": {"empathy": 5, "mental_resilience": 5}
                }
            },
            "battle_partners": {
                "requirements": {
                    "trust": [70, None],
                    "respect": [85, None],
                    "shared_trials": [10, None]
                },
                "description": "Forged in conflict, unbreakable in crisis",
                "rewards": {
                    "abilities": ["Back-to-Back", "Crisis Sync"],
                    "stat_bonuses": {"strength": 3, "endurance": 3}
                }
            },
            # ... other archetypes
        }
    
    @classmethod
    def check_archetypes(cls, state: RelationshipState) -> Set[str]:
        """Check which archetypes currently apply"""
        active = set()
        archetypes = cls.get_archetypes()
        
        for archetype_name, archetype_data in archetypes.items():
            if cls._meets_requirements(state.dimensions, archetype_data["requirements"], state):
                active.add(archetype_name)
        
        return active
    
    @classmethod
    def _meets_requirements(cls, dimensions: RelationshipDimensions, 
                          requirements: Dict[str, List], 
                          state: RelationshipState) -> bool:
        """Check if dimensions meet archetype requirements"""
        for requirement, bounds in requirements.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                continue
                
            min_val, max_val = bounds
            
            # Custom metrics
            if requirement == "shared_trials":
                value = sum(1 for i in state.history.recent_interactions 
                          if i.get('type') == 'crisis')
            else:
                value = getattr(dimensions, requirement, 0)
            
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
        
        return True

# ========================================================================
# ASYNC EVENT GENERATION
# ========================================================================

class AsyncEventGenerator:
    """Generates events asynchronously to avoid blocking"""
    
    def __init__(self):
        self.event_queue = asyncio.Queue(maxsize=100)
        self.generation_tasks = set()
    
    async def check_and_queue_event(self, state: RelationshipState, 
                                   context: Dict[str, Any]):
        """Check if event should generate and queue it"""
        # Quick significance check
        if not self._should_generate_event(state):
            return
        
        # Create task for async generation
        task = asyncio.create_task(
            self._generate_event_async(state, context)
        )
        self.generation_tasks.add(task)
        task.add_done_callback(self.generation_tasks.discard)
    
    def _should_generate_event(self, state: RelationshipState) -> bool:
        """Quick check if event generation is warranted"""
        significance_threshold = config.get('thresholds.significance', 10)
        
        # Check recent changes
        if state.history.significant_snapshots:
            last_diff = state.history.significant_snapshots[-1].get('diff', {})
            if any(abs(v) >= significance_threshold for v in last_diff.values() 
                  if isinstance(v, (int, float))):
                return True
        
        # Check patterns
        if state.history.active_patterns:
            return True
        
        # Check momentum
        if state.momentum.get_magnitude() > config.get('thresholds.momentum_trigger', 15):
            return True
        
        return False
    
    async def _generate_event_async(self, state: RelationshipState, 
                                   context: Dict[str, Any]):
        """Generate event asynchronously"""
        try:
            event = await self._generate_event(state, context)
            if event:
                await self.event_queue.put({
                    'state_key': state.canonical_key,
                    'event': event,
                    'timestamp': datetime.now()
                })
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")
        except Exception as e:
            logger.error(f"Error generating event: {e}")
    
    async def _generate_event(self, state: RelationshipState, 
                            context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate appropriate event based on state"""
        dims = state.dimensions
        
        # High trust with high unresolved conflict
        if dims.trust > 70 and dims.unresolved_conflict > 60:
            return self._generate_moment_of_truth(state)
        
        # Extreme volatility
        elif dims.volatility > 80 and abs(dims.affection) > 60:
            return self._generate_emotional_whiplash(state)
        
        # Pattern-specific events
        elif "push_pull" in state.history.active_patterns:
            return self._generate_pattern_event(state, "push_pull")
        
        return None
    
    def _generate_moment_of_truth(self, state: RelationshipState) -> Dict[str, Any]:
        """Generate a moment of truth event"""
        return {
            "type": "moment_of_truth",
            "title": "The Unspoken Truth",
            "description": "Hidden tensions demand resolution",
            "choices": [
                {
                    "id": "confront",
                    "text": "Address the issues directly",
                    "potential_impacts": {
                        "unresolved_conflict": -30,
                        "volatility": 20,
                        "trust": 10
                    }
                },
                {
                    "id": "avoid",
                    "text": "Maintain the status quo",
                    "potential_impacts": {
                        "unresolved_conflict": 15,
                        "hidden_agendas": 10
                    }
                }
            ]
        }
    
    def _generate_emotional_whiplash(self, state: RelationshipState) -> Dict[str, Any]:
        """Generate emotional intensity event"""
        emotion_type = "passion" if state.dimensions.affection > 0 else "conflict"
        return {
            "type": "emotional_whiplash",
            "title": f"Overwhelming {emotion_type.title()}",
            "choices": [
                {
                    "id": "embrace",
                    "text": f"Embrace the {emotion_type}",
                    "potential_impacts": {
                        "volatility": 10,
                        "affection": 20 if emotion_type == "passion" else -20
                    }
                },
                {
                    "id": "boundaries",
                    "text": "Create emotional distance",
                    "potential_impacts": {
                        "volatility": -20,
                        "frequency": -10
                    }
                }
            ]
        }
    
    def _generate_pattern_event(self, state: RelationshipState, 
                              pattern: str) -> Dict[str, Any]:
        """Generate pattern-specific event"""
        pattern_events = {
            "push_pull": {
                "type": "pattern_crisis",
                "title": "Breaking the Cycle",
                "description": "The push-pull dynamic reaches a breaking point"
            },
            "growing_distance": {
                "type": "reconnection_opportunity",
                "title": "Drifting Apart",
                "description": "A chance to reconnect or let go"
            }
        }
        return pattern_events.get(pattern)
    
    async def get_next_event(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get next event from queue (non-blocking)"""
        try:
            return await asyncio.wait_for(self.event_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

# Global event generator
event_generator = AsyncEventGenerator()

# ========================================================================
# OPTIMIZED MANAGER
# ========================================================================

class OptimizedRelationshipManager:
    """Performance-optimized relationship manager"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._memory_system = None
        self._lore_system = None
        
        # Enhanced cache with version tracking
        self._states_cache: Dict[str, Tuple[RelationshipState, datetime]] = {}
        self._cache_ttl = timedelta(seconds=config.get('thresholds.cache_ttl_seconds', 300))
        
        # Batch update queue
        self._update_queue = []
        self._last_batch_time = datetime.now()
    
    async def _get_memory_system(self) -> MemorySystem:
        """Lazy load memory system"""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(
                self.user_id, self.conversation_id
            )
        return self._memory_system
    
    async def _get_lore_system(self) -> LoreSystem:
        """Lazy load lore system"""
        if self._lore_system is None:
            self._lore_system = await LoreSystem.get_instance(
                self.user_id, self.conversation_id
            )
        return self._lore_system
    
    async def get_relationship_state(self, 
                                   entity1_type: str, entity1_id: int,
                                   entity2_type: str, entity2_id: int,
                                   skip_cache: bool = False) -> RelationshipState:
        """Get or create relationship state with version checking"""
        
        # Canonical ordering
        state = RelationshipState(
            entity1_type=entity1_type,
            entity1_id=entity1_id,
            entity2_type=entity2_type,
            entity2_id=entity2_id
        )
        cache_key = state.canonical_key
        
        # Check cache with version
        if not skip_cache and cache_key in self._states_cache:
            cached_state, cache_time = self._states_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                # Verify version
                if await self._check_version(cached_state):
                    return cached_state
        
        # Load from database
        state = await self._load_from_database(state)
        
        # Update patterns
        state.history.active_patterns = RelationshipPatternDetector.detect_patterns(
            state.history
        )
        state.active_archetypes = RelationshipArchetypes.check_archetypes(state)
        
        # Cache with timestamp
        self._states_cache[cache_key] = (state, datetime.now())
        
        return state
    
    async def _check_version(self, state: RelationshipState) -> bool:
        """Check if cached version is still valid"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT version FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND link_id = (
                    SELECT link_id FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                    AND canonical_key = $3
                    LIMIT 1
                )
            """, self.user_id, self.conversation_id, state.canonical_key)
            
            if row and row['version'] == state.version:
                return True
        
        return False
    
    async def _load_from_database(self, state: RelationshipState) -> RelationshipState:
        """Load state from database"""
        async with get_db_connection_context() as conn:
            # Use canonical key for lookup
            row = await conn.fetchrow("""
                SELECT link_id, dynamics, momentum, contexts, 
                       patterns, archetypes, version,
                       last_interaction, created_at
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND canonical_key = $3
            """, self.user_id, self.conversation_id, state.canonical_key)
            
            if row:
                # Parse stored data
                if row['dynamics']:
                    state.dimensions = RelationshipDimensions.from_dict(
                        json.loads(row['dynamics']) if isinstance(row['dynamics'], str) 
                        else row['dynamics']
                    )
                
                if row['momentum']:
                    momentum_data = json.loads(row['momentum']) if isinstance(row['momentum'], str) else row['momentum']
                    state.momentum.velocities = momentum_data.get('velocities', {})
                    state.momentum.inertia = momentum_data.get('inertia', 50.0)
                
                if row['patterns']:
                    state.history.active_patterns = set(json.loads(row['patterns']) if isinstance(row['patterns'], str) else row['patterns'])
                
                if row['archetypes']:
                    state.active_archetypes = set(json.loads(row['archetypes']) if isinstance(row['archetypes'], str) else row['archetypes'])
                
                state.version = row['version']
                state.created_at = row['created_at']
                state.last_interaction = row['last_interaction']
            else:
                # Create new relationship
                await self._create_new_relationship(state)
        
        return state
    
    async def _create_new_relationship(self, state: RelationshipState):
        """Create new relationship with canonical ordering"""
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        async with get_db_connection_context() as conn:
            # Set initial dynamics based on entity types
            if state.entity1_type == "player" or state.entity2_type == "player":
                state.dimensions.trust = 30
                state.dimensions.fascination = 40
                state.dimensions.frequency = 20
            
            # Create with canon
            await canon.find_or_create_social_link(
                ctx, conn,
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                entity1_type=state.entity1_type,
                entity1_id=state.entity1_id,
                entity2_type=state.entity2_type,
                entity2_id=state.entity2_id,
                link_type="neutral",
                link_level=0,
                link_history=[],
                dynamics=state.dimensions.to_dict(),
                experienced_crossroads=[],
                experienced_rituals=[],
                canonical_key=state.canonical_key
            )
    
    async def process_interaction(self,
                                entity1_type: str, entity1_id: int,
                                entity2_type: str, entity2_id: int,
                                interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process interaction with optimized flow"""
        
        # Get state
        state = await self.get_relationship_state(
            entity1_type, entity1_id, entity2_type, entity2_id
        )
        
        # Snapshot for comparison
        old_dimensions = RelationshipDimensions(**asdict(state.dimensions))
        
        # Calculate and apply impacts
        impacts = self._calculate_impacts(interaction, state)
        
        for dimension, change in impacts.items():
            if hasattr(state.dimensions, dimension):
                current = getattr(state.dimensions, dimension)
                setattr(state.dimensions, dimension, current + change)
                
                # Update momentum for significant changes
                if abs(change) > 3:
                    state.momentum.update_velocity(dimension, change)
        
        # Single clamp at the end
        state.dimensions.clamp()
        
        # Update history
        state.history.add_interaction(interaction.get('type', 'unknown'), 
                                    interaction.get('context', 'casual'))
        state.history.add_snapshot_if_significant(state.dimensions, old_dimensions)
        
        # Detect patterns
        new_patterns = RelationshipPatternDetector.detect_patterns(state.history)
        pattern_changes = new_patterns - state.history.active_patterns
        state.history.active_patterns = new_patterns
        
        # Check archetypes
        new_archetypes = RelationshipArchetypes.check_archetypes(state)
        archetype_changes = new_archetypes - state.active_archetypes
        state.active_archetypes = new_archetypes
        
        # Queue update for batching
        await self._queue_update(state)
        
        # Queue event generation (async)
        await event_generator.check_and_queue_event(state, {'interaction': interaction})
        
        # Return lightweight response
        return {
            "success": True,
            "impacts": impacts,
            "new_patterns": list(pattern_changes),
            "new_archetypes": list(archetype_changes),
            "dimensions_diff": state.dimensions.diff(old_dimensions)
        }
    
    def _calculate_impacts(self, interaction: Dict[str, Any], 
                         state: RelationshipState) -> Dict[str, float]:
        """Simplified impact calculation"""
        base_impacts = {
            "helpful_action": {"trust": 5, "respect": 3},
            "betrayal": {"trust": -30, "respect": -20, "unresolved_conflict": 25},
            "genuine_compliment": {"affection": 5, "respect": 3},
            "vulnerability_shared": {"trust": 8, "intimacy": 10},
            # ... other impacts
        }
        
        impacts = base_impacts.get(interaction.get('type', 'unknown'), {})
        
        # Apply state modifiers
        if state.dimensions.trust < 30:
            # Low trust amplifies negative
            impacts = {k: v * 1.5 if v < 0 else v for k, v in impacts.items()}
        
        return impacts
    
    async def _queue_update(self, state: RelationshipState):
        """Queue state update for batching"""
        self._update_queue.append(state)
        
        # Batch every second or every 10 updates
        if (len(self._update_queue) >= 10 or 
            (datetime.now() - self._last_batch_time).seconds >= 1):
            await self._flush_updates()
    
    async def _flush_updates(self):
        """Batch update all queued states"""
        if not self._update_queue:
            return
        
        updates_to_apply = self._update_queue[:]
        self._update_queue.clear()
        self._last_batch_time = datetime.now()
        
        # Build batch update data
        update_data = []
        for state in updates_to_apply:
            state.version += 1
            update_data.append({
                'canonical_key': state.canonical_key,
                'dynamics': json.dumps(state.dimensions.to_dict()),
                'momentum': json.dumps({
                    'velocities': state.momentum.velocities,
                    'inertia': state.momentum.inertia
                }),
                'patterns': json.dumps(list(state.history.active_patterns)),
                'archetypes': json.dumps(list(state.active_archetypes)),
                'version': state.version,
                'last_interaction': datetime.now()
            })
        
        # Batch update via LoreSystem
        lore_system = await self._get_lore_system()
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        # Use a single transaction for all updates
        async with get_db_connection_context() as conn:
            for data in update_data:
                await conn.execute("""
                    UPDATE SocialLinks
                    SET dynamics = $1::jsonb,
                        momentum = $2::jsonb,
                        patterns = $3::jsonb,
                        archetypes = $4::jsonb,
                        version = $5,
                        last_interaction = $6
                    WHERE user_id = $7 
                    AND conversation_id = $8
                    AND canonical_key = $9
                """, data['dynamics'], data['momentum'], data['patterns'],
                    data['archetypes'], data['version'], data['last_interaction'],
                    self.user_id, self.conversation_id, data['canonical_key'])
    
    async def apply_daily_drift(self):
        """Apply drift once per day instead of per-tick"""
        async with get_db_connection_context() as conn:
            # Get all relationships that need drift
            rows = await conn.fetch("""
                SELECT canonical_key, last_interaction, dynamics
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND last_interaction < NOW() - INTERVAL '1 day'
            """, self.user_id, self.conversation_id)
            
            drift_updates = []
            
            for row in rows:
                days_elapsed = (datetime.now() - row['last_interaction']).days
                if days_elapsed > 0:
                    dims = RelationshipDimensions.from_dict(
                        json.loads(row['dynamics']) if isinstance(row['dynamics'], str)
                        else row['dynamics']
                    )
                    
                    # Apply daily decay rates
                    freq_decay = config.get('decay_rates.frequency_daily', 0.05)
                    intimacy_decay = config.get('decay_rates.intimacy_daily', 0.02)
                    fascination_decay = config.get('decay_rates.fascination_daily', 0.01)
                    
                    dims.frequency *= (1 - freq_decay * days_elapsed)
                    dims.intimacy *= (1 - intimacy_decay * days_elapsed)
                    dims.fascination *= (1 - fascination_decay * days_elapsed)
                    
                    dims.clamp()
                    
                    drift_updates.append({
                        'canonical_key': row['canonical_key'],
                        'dynamics': json.dumps(dims.to_dict())
                    })
            
            # Batch update drift
            if drift_updates:
                for update in drift_updates:
                    await conn.execute("""
                        UPDATE SocialLinks
                        SET dynamics = $1::jsonb
                        WHERE user_id = $2 
                        AND conversation_id = $3
                        AND canonical_key = $4
                    """, update['dynamics'], self.user_id, 
                        self.conversation_id, update['canonical_key'])

# ========================================================================
# STREAMLINED FUNCTION TOOLS
# ========================================================================

@function_tool
async def process_relationship_interaction_tool(
    ctx: RunContextWrapper,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    interaction_type: str,
    context: str = "casual"
) -> dict:
    """Process interaction - returns only changes, not full state"""
    manager = OptimizedRelationshipManager(ctx.user_id, ctx.conversation_id)
    
    interaction = {
        "type": interaction_type,
        "context": context
    }
    
    result = await manager.process_interaction(
        entity1_type, entity1_id, entity2_type, entity2_id, interaction
    )
    
    # Check for events
    event = await event_generator.get_next_event()
    if event:
        result['event'] = event['event']
    
    return result

@function_tool
async def get_relationship_summary_tool(
    ctx: RunContextWrapper,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> dict:
    """Get lightweight relationship summary"""
    manager = OptimizedRelationshipManager(ctx.user_id, ctx.conversation_id)
    
    state = await manager.get_relationship_state(
        entity1_type, entity1_id, entity2_type, entity2_id
    )
    
    return state.to_summary()

@function_tool
async def apply_daily_drift_tool(ctx: RunContextWrapper) -> dict:
    """Apply daily relationship drift"""
    manager = OptimizedRelationshipManager(ctx.user_id, ctx.conversation_id)
    await manager.apply_daily_drift()
    
    # Also flush any pending updates
    await manager._flush_updates()
    
    return {"success": True, "message": "Daily drift applied"}

# ========================================================================
# OPTIMIZED ORCHESTRATOR
# ========================================================================

OptimizedRelationshipOrchestrator = Agent(
    name="OptimizedRelationshipOrchestrator",
    instructions="""You manage the optimized dynamic relationship system.
    
    Key optimizations:
    - Lightweight responses (only diffs, not full state)
    - Daily drift instead of per-tick decay
    - Async event generation
    - Batched database updates
    - Configurable thresholds
    
    Use tools efficiently:
    - process_interaction returns only changes
    - get_summary for full state when needed
    - apply_drift once per day
    
    Focus on meaningful interactions over constant updates.""",
    model="gpt-4",
    model_settings=ModelSettings(temperature=0.7),
    tools=[
        process_relationship_interaction_tool,
        get_relationship_summary_tool,
        apply_daily_drift_tool
    ]
)

# ========================================================================
# DATABASE SCHEMA UPDATE
# ========================================================================

"""
Required database schema changes:

ALTER TABLE SocialLinks ADD COLUMN canonical_key VARCHAR(255);
ALTER TABLE SocialLinks ADD COLUMN version INTEGER DEFAULT 0;
ALTER TABLE SocialLinks ADD COLUMN momentum JSONB;
ALTER TABLE SocialLinks ADD COLUMN patterns JSONB;
ALTER TABLE SocialLinks ADD COLUMN archetypes JSONB;

CREATE INDEX idx_social_links_canonical ON SocialLinks(user_id, conversation_id, canonical_key);
CREATE INDEX idx_social_links_drift ON SocialLinks(user_id, conversation_id, last_interaction);

-- Update existing records with canonical keys
UPDATE SocialLinks 
SET canonical_key = CASE 
    WHEN (entity1_type, entity1_id) <= (entity2_type, entity2_id)
    THEN entity1_type || '_' || entity1_id || '_' || entity2_type || '_' || entity2_id
    ELSE entity2_type || '_' || entity2_id || '_' || entity1_type || '_' || entity1_id
END;
"""

# ========================================================================
# EXPORT
# ========================================================================

__all__ = [
    'OptimizedRelationshipManager',
    'OptimizedRelationshipOrchestrator',
    'RelationshipConfig',
    'RelationshipState',
    'RelationshipDimensions'
]
