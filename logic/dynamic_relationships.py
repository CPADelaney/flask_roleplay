# logic/dynamic_relationships.py
"""
Dynamic Relationships System - Production-Ready Version v2
Fixes SQL parameterization, uses link_id for updates, and adds comprehensive validation.
"""

import json
import logging
import random
import asyncio
import os
import math
import statistics
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path
from pydantic import BaseModel, Field

import asyncpg

# Import core systems
from lore.core import canon
from lore.core.lore_system import LoreSystem
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

class DimensionDeltas(BaseModel):
    """Model for dimension deltas in relationship context updates"""
    trust: Optional[float] = Field(None, ge=-100, le=100)
    respect: Optional[float] = Field(None, ge=-100, le=100)
    affection: Optional[float] = Field(None, ge=-100, le=100)
    fascination: Optional[float] = Field(None, ge=-100, le=100)
    influence: Optional[float] = Field(None, ge=-100, le=100)
    dependence: Optional[float] = Field(None, ge=0, le=100)
    intimacy: Optional[float] = Field(None, ge=0, le=100)
    frequency: Optional[float] = Field(None, ge=0, le=100)
    volatility: Optional[float] = Field(None, ge=0, le=100)
    unresolved_conflict: Optional[float] = Field(None, ge=0, le=100)
    hidden_agendas: Optional[float] = Field(None, ge=0, le=100)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.dict().items() if v is not None}

# ========================================================================
# CONFIGURATION SYSTEM (ENHANCED VALIDATION)
# ========================================================================

class RelationshipConfig:
    """Centralized configuration for tunable parameters"""
    
    _instance = None
    _config = None
    
    # Define validation rules for all config values
    VALIDATION_RULES = {
        "thresholds.significance": (0, 100, float),
        "thresholds.momentum_trigger": (0, 50, float),
        "thresholds.decay_start_hours": (0, 168, int),  # Max 1 week
        "thresholds.cache_ttl_seconds": (60, 3600, int),  # 1 min to 1 hour
        "thresholds.decay_floor": (0, 1, float),
        "decay_rates.frequency_daily": (0, 1, float),
        "decay_rates.intimacy_daily": (0, 1, float),
        "decay_rates.fascination_daily": (0, 1, float),
        "velocity_factors.scale_base": (0, 1, float),
        "velocity_factors.scale_exponential": (None, None, bool),
        "velocity_factors.decay": (0, 1, float),
        "velocity_factors.max": (1, 100, float),
        "pattern_thresholds.push_pull_alternations": (0, 1, float),
        "pattern_thresholds.slow_burn_consistency": (3, 20, int),
        "pattern_thresholds.explosive_chemistry_affection": (0, 100, float),
        "pattern_thresholds.explosive_chemistry_volatility": (0, 100, float),
        "pattern_thresholds.frenemies_conflict": (0, 100, float),
        "pattern_thresholds.frenemies_respect": (-100, 100, float),
        "pattern_thresholds.rollercoaster_variance": (0, 10000, float),
        "event_queue.max_size": (10, 1000, int),
        "event_queue.log_drops": (None, None, bool),
        "event_queue.drop_log_interval": (1, 100, int),
    }
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_config()
        return cls._instance
    
    def load_config(self, config_path: str = None):
        """Load configuration from YAML file with env var override"""
        # Allow env var override
        if config_path is None:
            config_path = os.environ.get('RELATIONSHIP_CONFIG_PATH', 'config/relationships.yaml')
            
        config_file = Path(config_path)
        
        # Default configuration with all required values
        self._config = {
            "thresholds": {
                "significance": 10,
                "momentum_trigger": 8,
                "decay_start_hours": 24,
                "cache_ttl_seconds": 300,
                "decay_floor": 0.5,
            },
            "decay_rates": {
                "frequency_daily": 0.05,
                "intimacy_daily": 0.02,
                "fascination_daily": 0.01,
            },
            "velocity_factors": {
                "scale_base": 0.05,
                "scale_exponential": True,
                "decay": 0.9,
                "max": 10.0,
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
                "soulmates": {
                    "trust": [80, None],
                    "affection": [80, None],
                    "intimacy": [85, None]
                },
                "battle_partners": {
                    "trust": [70, None],
                    "respect": [85, None],
                    "shared_trials": [10, None]
                },
            },
            "event_queue": {
                "max_size": 100,
                "log_drops": True,
                "drop_log_interval": 10,
            }
        }
        
        # Load from file if exists
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    self._merge_config(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Validate all config values
        self._validate_config()
    
    def _merge_config(self, loaded: Dict):
        """Merge loaded config with defaults"""
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        deep_merge(self._config, loaded)
    
    def _validate_config(self):
        """Comprehensively validate all config values"""
        for path, (min_val, max_val, expected_type) in self.VALIDATION_RULES.items():
            value = self.get(path)
            if value is None:
                continue
                
            # Type check - accept both int and float for numeric types
            if expected_type in (int, float):
                if not isinstance(value, (int, float)):
                    logger.warning(f"Config {path}={value} is not numeric, using default")
                    continue
                # Convert to expected type
                if expected_type == float and isinstance(value, int):
                    value = float(value)
                    # Update the config with converted value
                    parts = path.split('.')
                    target = self._config
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = value
            elif expected_type and not isinstance(value, expected_type):
                logger.warning(f"Config {path}={value} is not {expected_type}, using default")
                continue
            
            # Range check for numeric types
            if expected_type in (int, float) and min_val is not None:
                if value < min_val or value > max_val:
                    clamped = max(min_val, min(max_val, value))
                    logger.warning(f"Config {path}={value} outside [{min_val}, {max_val}], clamping to {clamped}")
                    # Set the clamped value back
                    parts = path.split('.')
                    target = self._config
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = clamped
    
    def get(self, path: str, default=None, cast=None):
        """Get config value by dot notation path with type casting"""
        parts = path.split('.')
        value = self._config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        # Apply casting if requested
        if cast is not None and value is not None:
            try:
                return cast(value)
            except (ValueError, TypeError):
                logger.warning(f"Failed to cast config value {path}={value} to {cast}")
                return default
        
        return value

# Global config instance
config = RelationshipConfig.get_instance()

# ========================================================================
# SQL HELPERS
# ========================================================================

def generate_placeholders(count: int, start: int = 1) -> str:
    """Generate PostgreSQL placeholders like $1, $2, ..."""
    return ', '.join(f'${i}' for i in range(start, start + count))

def build_batch_update_query(table: str, key_columns: List[str], 
                           update_columns: List[str], num_rows: int) -> str:
    """Build a batch UPDATE query with proper placeholders"""
    # Calculate placeholders
    cols_per_row = len(key_columns) + len(update_columns)
    
    # Build VALUES rows
    value_rows = []
    for row_idx in range(num_rows):
        start_idx = row_idx * cols_per_row + 1
        placeholders = generate_placeholders(cols_per_row, start_idx)
        value_rows.append(f"({placeholders})")
    
    # Build column lists
    all_columns = key_columns + update_columns
    column_list = ', '.join(all_columns)
    
    # Build SET clause
    set_clauses = [f"{col} = v.{col}" for col in update_columns]
    set_clause = ', '.join(set_clauses)
    
    # Build WHERE clause
    where_clauses = [f"t.{col} = v.{col}" for col in key_columns]
    where_clause = ' AND '.join(where_clauses)
    
    query = f"""
        UPDATE {table} AS t
        SET {set_clause}
        FROM (VALUES {', '.join(value_rows)}) AS v({column_list})
        WHERE {where_clause}
    """
    
    return query

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
    influence: float = 0.0
    
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
        return {k: v for k, v in asdict(self).items() if v != 0.0}
    
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
            if abs(new_val - old_val) > 0.1:
                diff[attr] = new_val - old_val
        return diff

@dataclass
class RelationshipMomentum:
    """Tracks the velocity and direction of relationship changes"""
    velocities: Dict[str, float] = field(default_factory=dict)
    inertia: float = 50.0
    
    def get_velocity(self, dimension: str) -> float:
        """Get velocity for a dimension"""
        return self.velocities.get(dimension, 0.0)
    
    def update_velocity(self, dimension: str, change: float):
        """Update velocity with exponential back-off scaling"""
        decay = config.get('velocity_factors.decay', 0.9, float)
        max_velocity = config.get('velocity_factors.max', 10.0, float)
        use_exponential = config.get('velocity_factors.scale_exponential', True, bool)
        scale_base = config.get('velocity_factors.scale_base', 0.05, float)
        
        # Calculate scale with exponential back-off
        if use_exponential:
            scale = min(scale_base, abs(change) / 1000)
        else:
            scale = scale_base
        
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
        
        if abs(delta) < 0.1:
            self.context_deltas[situation].pop(dimension, None)
        else:
            self.context_deltas[situation][dimension] = delta
        
        if not self.context_deltas[situation]:
            self.context_deltas.pop(situation, None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'base_dimensions': self.base_dimensions.to_dict(),
            'context_deltas': self.context_deltas
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompactRelationshipContext':
        """Create from dictionary"""
        ctx = cls()
        if 'base_dimensions' in data:
            ctx.base_dimensions = RelationshipDimensions.from_dict(data['base_dimensions'])
        if 'context_deltas' in data:
            ctx.context_deltas = data['context_deltas']
        return ctx

@dataclass
class CompactRelationshipHistory:
    """Optimized history storage"""
    significant_snapshots: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_interactions: deque = field(default_factory=lambda: deque(maxlen=50))
    active_patterns: Set[str] = field(default_factory=set)
    pattern_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def add_snapshot_if_significant(self, dimensions: RelationshipDimensions, 
                                   previous: Optional[RelationshipDimensions] = None):
        """Only store snapshots with significant changes"""
        if previous is None and self.significant_snapshots:
            last = RelationshipDimensions.from_dict(
                self.significant_snapshots[-1]['dimensions']
            )
            diff = dimensions.diff(last)
        elif previous:
            diff = dimensions.diff(previous)
        else:
            diff = {'initial': True}
        
        # Check if significant
        significance_threshold = config.get('thresholds.significance', 10, float)
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
    """Optimized relationship state with link_id tracking"""
    # Entity identifiers
    entity1_type: str
    entity1_id: int
    entity2_type: str
    entity2_id: int
    
    # Database tracking
    link_id: Optional[int] = None
    
    @property
    def canonical_key(self) -> str:
        """Get canonical key for this relationship"""
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
    history: CompactRelationshipHistory = field(default_factory=lambda: CompactRelationshipHistory())
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    version: int = 0
    
    # Computed properties
    active_archetypes: Set[str] = field(default_factory=set)
    
    def get_duration_days(self) -> int:
        """How long this relationship has existed"""
        return (datetime.now() - self.created_at).days
    
    def to_summary(self) -> Dict[str, Any]:
        """Get lightweight summary for agent tools"""
        return {
            'link_id': self.link_id,
            'version': self.version,
            'dimensions': self.dimensions.to_dict(),
            'momentum_magnitude': self.momentum.get_magnitude(),
            'patterns': list(self.history.active_patterns),
            'archetypes': list(self.active_archetypes),
            'duration_days': self.get_duration_days()
        }

# ========================================================================
# PATTERN DETECTION
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
            if 'diff' in snapshots[i] and 'intimacy' in snapshots[i]['diff']:
                intimacy_changes.append(snapshots[i]['diff']['intimacy'])
        
        if len(intimacy_changes) < 4:
            return False
        
        # Look for alternating signs
        alternations = 0
        for i in range(1, len(intimacy_changes)):
            if intimacy_changes[i] * intimacy_changes[i-1] < 0:
                alternations += 1
        
        threshold = config.get('pattern_thresholds.push_pull_alternations', 0.4, float)
        return alternations >= len(intimacy_changes) * threshold
    
    @staticmethod
    def _detect_slow_burn(snapshots: List[Dict]) -> bool:
        """Gradual consistent improvement"""
        consistency_req = config.get('pattern_thresholds.slow_burn_consistency', 5, int)
        if len(snapshots) < consistency_req:
            return False
        
        # Check trust trajectory
        trust_values = [s['dimensions'].get('trust', 0) for s in snapshots[-consistency_req:]]
        
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
        
        affection_threshold = config.get('pattern_thresholds.explosive_chemistry_affection', 70, float)
        volatility_threshold = config.get('pattern_thresholds.explosive_chemistry_volatility', 60, float)
        
        return avg_affection > affection_threshold and avg_volatility > volatility_threshold
    
    @staticmethod
    def _detect_frenemies(snapshots: List[Dict]) -> bool:
        """High conflict but also high respect"""
        if len(snapshots) < 3:
            return False
        
        recent = snapshots[-3:]
        avg_conflict = statistics.mean(s['dimensions'].get('unresolved_conflict', 0) for s in recent)
        avg_respect = statistics.mean(s['dimensions'].get('respect', 0) for s in recent)
        
        conflict_threshold = config.get('pattern_thresholds.frenemies_conflict', 50, float)
        respect_threshold = config.get('pattern_thresholds.frenemies_respect', 60, float)
        
        return avg_conflict > conflict_threshold and avg_respect > respect_threshold
    
    @staticmethod
    def _detect_emotional_rollercoaster(snapshots: List[Dict]) -> bool:
        """Wild swings in emotional dimensions"""
        if len(snapshots) < 5:
            return False
        
        affection_values = [s['dimensions'].get('affection', 0) for s in snapshots[-5:]]
        
        # Calculate variance
        if len(set(affection_values)) == 1:
            return False
        
        variance = statistics.variance(affection_values)
        threshold = config.get('pattern_thresholds.rollercoaster_variance', 400, float)
        
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
# ARCHETYPE SYSTEM
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
            "mentor_student": {
                "requirements": {
                    "respect": [75, None],
                    "influence": [-30, -70],
                    "trust": [60, None]
                },
                "description": "One guides, one learns, both grow",
                "rewards": {
                    "abilities": ["Teachings", "Quick Learner"],
                    "stat_bonuses": {"intelligence": 4, "wisdom": 2}
                }
            },
            "rivals": {
                "requirements": {
                    "respect": [60, None],
                    "affection": [-20, 20],
                    "volatility": [40, None]
                },
                "description": "Competition drives excellence",
                "rewards": {
                    "abilities": ["Competitive Edge", "Never Back Down"],
                    "stat_bonuses": {"determination": 5, "agility": 3}
                }
            },
            "toxic_bond": {
                "requirements": {
                    "dependence": [70, None],
                    "trust": [None, 30],
                    "unresolved_conflict": [60, None]
                },
                "description": "Destructive but inescapable connection",
                "rewards": {
                    "abilities": ["Trauma Bond", "Emotional Vampire"],
                    "stat_bonuses": {"corruption": 10, "mental_resilience": -5}
                }
            }
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
    """Generates events asynchronously with better queue management"""
    
    def __init__(self):
        max_size = config.get('event_queue.max_size', 100, int)
        self.event_queue = asyncio.Queue(maxsize=max_size)
        self.generation_tasks = set()
        self.dropped_events = deque(maxlen=100)  # Track dropped events
        self.drop_counter = 0
    
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
        significance_threshold = config.get('thresholds.significance', 10, float)
        
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
        momentum_trigger = config.get('thresholds.momentum_trigger', 8, float)
        if state.momentum.get_magnitude() > momentum_trigger:
            return True
        
        return False
    
    async def _generate_event_async(self, state: RelationshipState, 
                                   context: Dict[str, Any]):
        """Generate event asynchronously"""
        try:
            event = await self._generate_event(state, context)
            if event:
                try:
                    await self.event_queue.put({
                        'state_key': state.canonical_key,
                        'event': event,
                        'timestamp': datetime.now()
                    })
                except asyncio.QueueFull:
                    # Log dropped event
                    self.drop_counter += 1
                    self.dropped_events.append({
                        'state_key': state.canonical_key,
                        'event_type': event.get('type', 'unknown'),
                        'timestamp': datetime.now()
                    })
                    
                    # Log periodically
                    log_interval = config.get('event_queue.drop_log_interval', 10, int)
                    if config.get('event_queue.log_drops', True) and self.drop_counter % log_interval == 0:
                        logger.warning(f"Event queue full - dropped {self.drop_counter} events. "
                                     f"Recent: {list(self.dropped_events)[-5:]}")
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
        
        elif "growing_distance" in state.history.active_patterns:
            return self._generate_pattern_event(state, "growing_distance")
        
        # Archetype events
        elif "toxic_bond" in state.active_archetypes:
            return self._generate_archetype_event(state, "toxic_bond")
        
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
                "description": "The push-pull dynamic reaches a breaking point",
                "choices": [
                    {
                        "id": "confront_pattern",
                        "text": "Call out the pattern directly",
                        "potential_impacts": {
                            "intimacy": -10,
                            "trust": 15,
                            "volatility": 25
                        }
                    },
                    {
                        "id": "play_along",
                        "text": "Continue the dance",
                        "potential_impacts": {
                            "intimacy": 5,
                            "volatility": 10,
                            "unresolved_conflict": 10
                        }
                    }
                ]
            },
            "growing_distance": {
                "type": "reconnection_opportunity",
                "title": "Drifting Apart",
                "description": "A chance to reconnect or let go",
                "choices": [
                    {
                        "id": "reach_out",
                        "text": "Make an effort to reconnect",
                        "potential_impacts": {
                            "frequency": 20,
                            "intimacy": 10,
                            "trust": 5
                        }
                    },
                    {
                        "id": "let_go",
                        "text": "Accept the growing distance",
                        "potential_impacts": {
                            "frequency": -20,
                            "intimacy": -15,
                            "dependence": -10
                        }
                    }
                ]
            }
        }
        return pattern_events.get(pattern)
    
    def _generate_archetype_event(self, state: RelationshipState,
                                archetype: str) -> Dict[str, Any]:
        """Generate archetype-specific event"""
        archetype_events = {
            "toxic_bond": {
                "type": "archetype_crisis",
                "title": "Breaking Point",
                "description": "The toxic nature of your bond becomes undeniable",
                "choices": [
                    {
                        "id": "break_free",
                        "text": "Attempt to break the bond",
                        "potential_impacts": {
                            "dependence": -20,
                            "unresolved_conflict": 30,
                            "volatility": 40
                        }
                    },
                    {
                        "id": "deeper_entanglement",
                        "text": "Surrender to the toxicity",
                        "potential_impacts": {
                            "dependence": 20,
                            "trust": -10,
                            "hidden_agendas": 15
                        }
                    }
                ]
            }
        }
        return archetype_events.get(archetype)
    
    async def get_next_event(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get next event from queue (non-blocking)"""
        try:
            return await asyncio.wait_for(self.event_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None
    
    async def drain_events(self, max_events: int = 100) -> List[Dict[str, Any]]:
        """Drain all pending events (for batch processing)"""
        events = []
        try:
            while len(events) < max_events and not self.event_queue.empty():
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.01)
                events.append(event)
        except asyncio.TimeoutError:
            pass
        return events

# Global event generator
event_generator = AsyncEventGenerator()

# ========================================================================
# OPTIMIZED MANAGER
# ========================================================================

class OptimizedRelationshipManager:
    """Performance-optimized relationship manager with fixed SQL"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._memory_system = None
        self._lore_system = None
        
        # Enhanced cache with version tracking
        self._states_cache: Dict[str, Tuple[RelationshipState, datetime]] = {}
        self._cache_ttl = timedelta(seconds=config.get('thresholds.cache_ttl_seconds', 300, int))
        
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
                # Use stored link_id for efficient version check
                if cached_state.link_id and await self._check_version_by_id(cached_state):
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
    
    async def _check_version_by_id(self, state: RelationshipState) -> bool:
        """Check if cached version is still valid using link_id"""
        if not state.link_id:
            return False
            
        async with get_db_connection_context() as conn:
            version = await conn.fetchval("""
                SELECT version FROM SocialLinks
                WHERE link_id = $1
            """, state.link_id)
            
            return version == state.version
    
    async def _load_from_database(self, state: RelationshipState) -> RelationshipState:
        """Load state from database including contexts"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT link_id, dynamics, momentum, contexts, 
                       patterns, archetypes, version,
                       last_interaction, created_at
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND canonical_key = $3
            """, self.user_id, self.conversation_id, state.canonical_key)
            
            if row:
                state.link_id = row['link_id']
                
                # Parse all data
                if row['dynamics']:
                    state.dimensions = RelationshipDimensions.from_dict(
                        json.loads(row['dynamics']) if isinstance(row['dynamics'], str) 
                        else row['dynamics']
                    )
                
                if row['momentum']:
                    momentum_data = json.loads(row['momentum']) if isinstance(row['momentum'], str) else row['momentum']
                    state.momentum.velocities = momentum_data.get('velocities', {})
                    state.momentum.inertia = momentum_data.get('inertia', 50.0)
                
                if row['contexts']:
                    contexts_data = json.loads(row['contexts']) if isinstance(row['contexts'], str) else row['contexts']
                    state.contexts = CompactRelationshipContext.from_dict(contexts_data)
                
                if row['patterns']:
                    state.history.active_patterns = set(json.loads(row['patterns']) if isinstance(row['patterns'], str) else row['patterns'])
                
                if row['archetypes']:
                    state.active_archetypes = set(json.loads(row['archetypes']) if isinstance(row['archetypes'], str) else row['archetypes'])
                
                state.version = row['version']
                state.created_at = row['created_at']
                state.last_interaction = row['last_interaction']
            else:
                await self._create_new_relationship(state)
        
        return state
    
    async def _create_new_relationship(self, state: RelationshipState):
        """Create new relationship WITHOUT using canon to avoid recursion"""
        import json
        from datetime import datetime
        
        async with get_db_connection_context() as conn:
            # Set initial dynamics based on entity types
            if state.entity1_type == "player" or state.entity2_type == "player":
                state.dimensions.trust = 30
                state.dimensions.fascination = 40
                state.dimensions.frequency = 20
            
            # FIXED: Insert directly instead of calling canon.find_or_create_social_link
            # This avoids the circular dependency
            try:
                link_id = await conn.fetchval("""
                    INSERT INTO SocialLinks (
                        user_id, conversation_id,
                        entity1_type, entity1_id, entity2_type, entity2_id,
                        canonical_key, dynamics, momentum, contexts,
                        patterns, archetypes, version,
                        last_interaction, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, 
                            $10::jsonb, $11::jsonb, $12::jsonb, $13, $14, $15)
                    ON CONFLICT (user_id, conversation_id, canonical_key) 
                    DO UPDATE SET 
                        last_interaction = EXCLUDED.last_interaction
                    RETURNING link_id
                """,
                    self.user_id, self.conversation_id,
                    state.entity1_type, state.entity1_id,
                    state.entity2_type, state.entity2_id,
                    state.canonical_key,
                    json.dumps(state.dimensions.to_dict()),
                    json.dumps({
                        'velocities': state.momentum.velocities,
                        'inertia': state.momentum.inertia
                    }),
                    json.dumps(state.contexts.to_dict()),
                    json.dumps(list(state.history.active_patterns)),
                    json.dumps(list(state.active_archetypes)),
                    0,  # version
                    datetime.now(),
                    datetime.now()
                )
            except asyncpg.UniqueViolationError:
                # Handle race condition - another process created it first
                row = await conn.fetchrow("""
                    SELECT link_id FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2 AND canonical_key = $3
                """, self.user_id, self.conversation_id, state.canonical_key)
                link_id = row['link_id'] if row else None
                
                if not link_id:
                    raise
            
            # Store link_id
            state.link_id = link_id
            
            # Log the creation (optional)
            logger.info(f"Created new relationship {state.canonical_key} with link_id {link_id}")
    
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
            "conflict_resolved": {"unresolved_conflict": -20, "trust": 5},
            "boundary_violated": {"trust": -15, "respect": -10, "volatility": 15},
            "support_provided": {"trust": 7, "dependence": 5, "affection": 3},
            "criticism_harsh": {"respect": -5, "affection": -8, "unresolved_conflict": 10},
            "shared_success": {"respect": 8, "affection": 5, "trust": 3},
            "deception_discovered": {"trust": -25, "hidden_agendas": 20, "volatility": 20}
        }
        
        impacts = base_impacts.get(interaction.get('type', 'unknown'), {})
        
        # Apply state modifiers
        if state.dimensions.trust < 30:
            # Low trust amplifies negative
            impacts = {k: v * 1.5 if v < 0 else v for k, v in impacts.items()}
        
        if state.dimensions.volatility > 70:
            # High volatility amplifies all changes
            impacts = {k: v * 1.3 for k, v in impacts.items()}
        
        return impacts
    
    async def _queue_update(self, state: RelationshipState):
        """Queue state update for batching"""
        self._update_queue.append(state)
        
        # Batch every second or every 10 updates
        if (len(self._update_queue) >= 10 or 
            (datetime.now() - self._last_batch_time).seconds >= 1):
            await self._flush_updates()
    
    async def _flush_updates(self):
        """Batch update all queued states using proper SQL"""
        if not self._update_queue:
            return
        
        updates_to_apply = self._update_queue[:]
        self._update_queue.clear()
        self._last_batch_time = datetime.now()
        
        # Build flattened values list
        flat_values = []
        for state in updates_to_apply:
            state.version += 1
            flat_values.extend([
                state.link_id,  # Use link_id instead of canonical_key
                json.dumps(state.dimensions.to_dict()),
                json.dumps({
                    'velocities': state.momentum.velocities,
                    'inertia': state.momentum.inertia
                }),
                json.dumps(state.contexts.to_dict()),  # Include contexts
                json.dumps(list(state.history.active_patterns)),
                json.dumps(list(state.active_archetypes)),
                state.version,
                datetime.now()
            ])
        
        # Build query
        num_rows = len(updates_to_apply)
        query = build_batch_update_query(
            'SocialLinks',
            key_columns=['link_id'],  # Use link_id as key
            update_columns=['dynamics', 'momentum', 'contexts', 'patterns', 
                          'archetypes', 'version', 'last_interaction'],
            num_rows=num_rows
        )
        
        # Execute with flattened values
        async with get_db_connection_context() as conn:
            await conn.execute(query, *flat_values)
    
    async def apply_daily_drift(self):
        """Apply drift once per day with proper batch SQL"""
        async with get_db_connection_context() as conn:
            # Get all relationships that need drift
            rows = await conn.fetch("""
                SELECT link_id, last_interaction, dynamics
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND last_interaction < NOW() - INTERVAL '1 day'
            """, self.user_id, self.conversation_id)
            
            if not rows:
                return
            
            # Process drift
            flat_values = []
            decay_floor = config.get('thresholds.decay_floor', 0.5, float)
            
            for row in rows:
                days_elapsed = (datetime.now() - row['last_interaction']).days
                if days_elapsed > 0:
                    dims = RelationshipDimensions.from_dict(
                        json.loads(row['dynamics']) if isinstance(row['dynamics'], str)
                        else row['dynamics']
                    )
                    
                    # Apply daily decay rates
                    freq_decay = config.get('decay_rates.frequency_daily', 0.05, float)
                    intimacy_decay = config.get('decay_rates.intimacy_daily', 0.02, float)
                    fascination_decay = config.get('decay_rates.fascination_daily', 0.01, float)
                    
                    # Apply decay with floor check
                    dims.frequency *= (1 - freq_decay * days_elapsed)
                    if dims.frequency < decay_floor:
                        dims.frequency = 0.0
                        
                    dims.intimacy *= (1 - intimacy_decay * days_elapsed)
                    if dims.intimacy < decay_floor:
                        dims.intimacy = 0.0
                        
                    dims.fascination *= (1 - fascination_decay * days_elapsed)
                    if dims.fascination < decay_floor:
                        dims.fascination = 0.0
                    
                    dims.clamp()
                    
                    flat_values.extend([
                        row['link_id'],
                        json.dumps(dims.to_dict())
                    ])
            
            if flat_values:
                # Build and execute batch update
                num_rows = len(flat_values) // 2
                query = build_batch_update_query(
                    'SocialLinks',
                    key_columns=['link_id'],
                    update_columns=['dynamics'],
                    num_rows=num_rows
                )
                await conn.execute(query, *flat_values)

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
    context: str = "casual",
    check_for_event: bool = True
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
    
    # Only check for events if requested
    if check_for_event:
        event = await event_generator.get_next_event()
        if event:
            result['event'] = event['event']
    
    return result

@function_tool
async def poll_relationship_events_tool(
    ctx: RunContextWrapper,
    timeout: float = 0.1
) -> dict:
    """Poll for pending relationship events"""
    event = await event_generator.get_next_event(timeout)
    if event:
        return {
            "has_event": True,
            "event": event['event'],
            "state_key": event['state_key']
        }
    return {"has_event": False}

@function_tool
async def drain_relationship_events_tool(
    ctx: RunContextWrapper,
    max_events: int = 50
) -> dict:
    """Drain all pending events for batch processing"""
    events = await event_generator.drain_events(max_events)
    return {
        "events": events,
        "count": len(events)
    }

@function_tool
async def get_relationship_summary_tool(
    ctx: RunContextWrapper,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> dict:
    """Get lightweight relationship summary with version"""
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

@function_tool
async def update_relationship_context_tool(
    ctx: RunContextWrapper,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    situation: str,
    dimension_deltas: DimensionDeltas
) -> dict:
    """Update context-specific relationship modifiers"""
    manager = OptimizedRelationshipManager(ctx.user_id, ctx.conversation_id)
    
    state = await manager.get_relationship_state(
        entity1_type, entity1_id, entity2_type, entity2_id
    )
    
    # Convert to dict and update context deltas
    deltas_dict = dimension_deltas.to_dict()
    for dimension, delta in deltas_dict.items():
        state.contexts.set_context_delta(situation, dimension, delta)
    
    # Queue for update
    await manager._queue_update(state)
    
    return {
        "success": True,
        "situation": situation,
        "deltas_applied": deltas_dict
    }

# ========================================================================
# OPTIMIZED ORCHESTRATOR
# ========================================================================

OptimizedRelationshipOrchestrator = Agent(
    name="OptimizedRelationshipOrchestrator",
    instructions="""You manage the optimized dynamic relationship system.
    
    Key optimizations:
    - Lightweight responses (only diffs, not full state)
    - Daily drift instead of per-tick decay
    - Async event generation with separate polling
    - Batched database updates
    - Configurable thresholds
    - Context-aware relationships
    
    Use tools efficiently:
    - process_interaction returns only changes
    - poll_events separately when needed
    - drain_events for batch processing
    - get_summary includes version for caching
    - apply_drift once per day
    - update_context for situation-specific modifiers
    
    Focus on meaningful interactions over constant updates.
    
    Available interaction types:
    - helpful_action, betrayal, genuine_compliment
    - vulnerability_shared, conflict_resolved, boundary_violated
    - support_provided, criticism_harsh, shared_success
    - deception_discovered
    
    Watch for patterns:
    - push_pull, slow_burn, explosive_chemistry
    - frenemies, emotional_rollercoaster, growing_distance
    
    And archetypes:
    - soulmates, battle_partners, mentor_student
    - rivals, toxic_bond""",
    model="gpt-4",
    model_settings=ModelSettings(temperature=0.7),
    tools=[
        process_relationship_interaction_tool,
        get_relationship_summary_tool,
        apply_daily_drift_tool,
        poll_relationship_events_tool,
        drain_relationship_events_tool,
        update_relationship_context_tool
    ]
)

# ========================================================================
# DATABASE SCHEMA UPDATE
# ========================================================================

"""
Required database schema changes:

-- Add new columns
ALTER TABLE SocialLinks ADD COLUMN canonical_key VARCHAR(255);
ALTER TABLE SocialLinks ADD COLUMN version INTEGER DEFAULT 0;
ALTER TABLE SocialLinks ADD COLUMN momentum JSONB DEFAULT '{}';
ALTER TABLE SocialLinks ADD COLUMN contexts JSONB DEFAULT '{}';
ALTER TABLE SocialLinks ADD COLUMN patterns JSONB DEFAULT '[]';
ALTER TABLE SocialLinks ADD COLUMN archetypes JSONB DEFAULT '[]';

-- Add indexes
CREATE INDEX idx_social_links_canonical ON SocialLinks(user_id, conversation_id, canonical_key);
CREATE INDEX idx_social_links_drift ON SocialLinks(user_id, conversation_id, last_interaction);
CREATE INDEX idx_social_links_version ON SocialLinks(link_id, version);

-- Update existing records with canonical keys
UPDATE SocialLinks 
SET canonical_key = CASE 
    WHEN (entity1_type, entity1_id) <= (entity2_type, entity2_id)
    THEN entity1_type || '_' || entity1_id || '_' || entity2_type || '_' || entity2_id
    ELSE entity2_type || '_' || entity2_id || '_' || entity1_type || '_' || entity1_id
END
WHERE canonical_key IS NULL;

-- Ensure non-null for new deployments
ALTER TABLE SocialLinks ALTER COLUMN canonical_key SET NOT NULL;
"""

# ========================================================================
# EXPORT
# ========================================================================

__all__ = [
    'OptimizedRelationshipManager',
    'OptimizedRelationshipOrchestrator',
    'RelationshipConfig',
    'RelationshipState',
    'RelationshipDimensions',
    'RelationshipMomentum',
    'CompactRelationshipContext',
    'CompactRelationshipHistory',
    'RelationshipPatternDetector',
    'RelationshipArchetypes',
    'AsyncEventGenerator',
    'event_generator',
    'process_relationship_interaction_tool',
    'get_relationship_summary_tool',
    'apply_daily_drift_tool',
    'poll_relationship_events_tool',
    'drain_relationship_events_tool',
    'update_relationship_context_tool'
]
