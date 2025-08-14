# Tension Accumulation System

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import json

class TensionSource(Enum):
    """Types of tension sources in slice-of-life conflicts"""
    ROUTINE_VIOLATION = "routine_violation"
    UNMET_EXPECTATION = "unmet_expectation"
    RESOURCE_COMPETITION = "resource_competition"
    SOCIAL_SLIGHT = "social_slight"
    PATTERN_FRICTION = "pattern_friction"
    ACCUMULATED_IRRITATION = "accumulated_irritation"
    POWER_CHALLENGE = "power_challenge"
    ATTENTION_IMBALANCE = "attention_imbalance"

class FrictionPoint:
    """Represents a single point of friction between characters"""
    def __init__(self, source: TensionSource, intensity: float, 
                 participants: Set[str], context: Dict):
        self.source = source
        self.intensity = intensity  # 0.0 to 1.0
        self.participants = participants
        self.context = context
        self.timestamp = datetime.now()
        self.decay_rate = 0.05  # How quickly it fades if not reinforced
        
    def decay(self, hours_passed: float) -> float:
        """Natural decay of friction over time"""
        self.intensity *= (1 - self.decay_rate * hours_passed)
        return self.intensity

@dataclass
class TensionAccumulator:
    """Tracks building tension between characters"""
    character_a: str
    character_b: str
    current_pressure: float = 0.0
    friction_points: List[FrictionPoint] = field(default_factory=list)
    micro_aggressions: List[Dict] = field(default_factory=list)
    boiling_point: float = 10.0  # Threshold for conflict emergence
    pressure_relief_rate: float = 0.1  # Natural pressure dissipation
    
    def add_friction(self, friction: FrictionPoint):
        """Add a new friction point and calculate pressure increase"""
        self.friction_points.append(friction)
        
        # Recent frictions have more impact
        recency_multiplier = 1.0
        if len(self.friction_points) > 1:
            time_since_last = (friction.timestamp - 
                             self.friction_points[-2].timestamp).total_seconds() / 3600
            if time_since_last < 1:  # Within an hour
                recency_multiplier = 1.5  # Compounds faster
                
        pressure_increase = friction.intensity * recency_multiplier
        self.current_pressure += pressure_increase
        
        # Check for pattern-based amplification
        if self._detect_pattern_amplification():
            self.current_pressure *= 1.2
            
        return self.current_pressure
    
    def add_micro_aggression(self, aggression: Dict):
        """Track small slights that build up over time"""
        self.micro_aggressions.append({
            **aggression,
            'timestamp': datetime.now(),
            'weight': aggression.get('weight', 0.1)
        })
        
        # Every 5 micro-aggressions increases pressure
        if len(self.micro_aggressions) % 5 == 0:
            self.current_pressure += 0.5
            
    def _detect_pattern_amplification(self) -> bool:
        """Check if similar frictions are repeating (pattern detection)"""
        if len(self.friction_points) < 3:
            return False
            
        recent = self.friction_points[-3:]
        sources = [f.source for f in recent]
        
        # If same source appears multiple times, it's a pattern
        return len(sources) != len(set(sources))
    
    def check_boiling_point(self) -> Optional[Dict]:
        """Determine if tension has reached breaking point"""
        if self.current_pressure >= self.boiling_point:
            # Analyze what type of conflict should emerge
            dominant_source = self._get_dominant_tension_source()
            
            return {
                'type': 'boiling_point_reached',
                'participants': [self.character_a, self.character_b],
                'pressure': self.current_pressure,
                'dominant_source': dominant_source,
                'trigger_suggestion': self._suggest_trigger(dominant_source),
                'intensity_level': self._calculate_intensity()
            }
        return None
    
    def _get_dominant_tension_source(self) -> TensionSource:
        """Identify the primary source of tension"""
        source_weights = {}
        for friction in self.friction_points[-10:]:  # Last 10 frictions
            source = friction.source
            source_weights[source] = source_weights.get(source, 0) + friction.intensity
            
        return max(source_weights, key=source_weights.get)
    
    def _suggest_trigger(self, source: TensionSource) -> str:
        """Suggest a naturalistic trigger for the conflict"""
        triggers = {
            TensionSource.ROUTINE_VIOLATION: "deviation from expected routine",
            TensionSource.UNMET_EXPECTATION: "forgotten promise or expectation",
            TensionSource.RESOURCE_COMPETITION: "both wanting the same thing",
            TensionSource.SOCIAL_SLIGHT: "perceived disrespect or dismissal",
            TensionSource.PATTERN_FRICTION: "the same issue coming up again",
            TensionSource.ACCUMULATED_IRRITATION: "one too many small annoyances",
            TensionSource.POWER_CHALLENGE: "subtle challenge to established dynamic",
            TensionSource.ATTENTION_IMBALANCE: "feeling ignored or overshadowed"
        }
        return triggers.get(source, "accumulated tension")
    
    def _calculate_intensity(self) -> str:
        """Determine conflict intensity based on pressure level"""
        ratio = self.current_pressure / self.boiling_point
        if ratio < 1.2:
            return "subtext"
        elif ratio < 1.5:
            return "tension"
        elif ratio < 2.0:
            return "passive_aggressive"
        else:
            return "confrontation"
    
    def apply_time_decay(self, hours_passed: float):
        """Apply natural pressure relief over time"""
        self.current_pressure *= (1 - self.pressure_relief_rate * hours_passed)
        self.current_pressure = max(0, self.current_pressure)
        
        # Decay individual friction points
        self.friction_points = [f for f in self.friction_points 
                               if f.decay(hours_passed) > 0.01]

class TensionAnalyzer:
    """Analyzes game state to detect and generate tension"""
    
    def __init__(self):
        self.accumulators: Dict[Tuple[str, str], TensionAccumulator] = {}
        self.global_modifiers = {
            'morning': 1.2,  # People more irritable in morning
            'evening': 0.8,   # More relaxed in evening
            'weekend': 0.7,   # Less pressure on weekends
            'hungry': 1.3,    # Hunger increases irritability
            'tired': 1.4,     # Fatigue lowers tolerance
        }
        
    def detect_friction_opportunity(self, scene_context: Dict) -> List[Dict]:
        """Analyze scene for potential friction points"""
        opportunities = []
        
        # Check for routine violations
        if scene_context.get('routine_broken'):
            opportunities.append({
                'source': TensionSource.ROUTINE_VIOLATION,
                'intensity': 0.3,
                'description': 'Expected routine was disrupted'
            })
            
        # Check for resource competition
        shared_resources = scene_context.get('shared_resources', [])
        if len(shared_resources) > 0:
            for resource in shared_resources:
                if resource['availability'] < resource['demand']:
                    opportunities.append({
                        'source': TensionSource.RESOURCE_COMPETITION,
                        'intensity': 0.4,
                        'description': f'Competition for {resource["name"]}'
                    })
                    
        # Check for attention imbalance
        attention_distribution = scene_context.get('attention_distribution', {})
        if attention_distribution:
            avg_attention = sum(attention_distribution.values()) / len(attention_distribution)
            for character, attention in attention_distribution.items():
                if attention < avg_attention * 0.5:
                    opportunities.append({
                        'source': TensionSource.ATTENTION_IMBALANCE,
                        'intensity': 0.2,
                        'affected': character,
                        'description': f'{character} receiving less attention'
                    })
                    
        return opportunities
    
    def calculate_group_tension_dynamics(self, characters: List[str], 
                                        context: Dict) -> Dict:
        """Calculate tension dynamics for a group"""
        tension_web = {}
        
        for i, char_a in enumerate(characters):
            for char_b in characters[i+1:]:
                key = tuple(sorted([char_a, char_b]))
                
                if key not in self.accumulators:
                    self.accumulators[key] = TensionAccumulator(
                        character_a=key[0],
                        character_b=key[1]
                    )
                    
                accumulator = self.accumulators[key]
                
                # Apply contextual modifiers
                modifier = 1.0
                for condition, mult in self.global_modifiers.items():
                    if context.get(condition):
                        modifier *= mult
                        
                accumulator.current_pressure *= modifier
                
                # Check for emergence
                emergence = accumulator.check_boiling_point()
                if emergence:
                    tension_web[key] = emergence
                    
        return tension_web
    
    def generate_micro_aggression(self, perpetrator: str, target: str, 
                                 context: Dict) -> Dict:
        """Generate contextual micro-aggression"""
        templates = [
            {'text': 'subtle eye roll', 'weight': 0.1},
            {'text': 'delayed response', 'weight': 0.15},
            {'text': 'talking over', 'weight': 0.2},
            {'text': 'dismissive tone', 'weight': 0.15},
            {'text': 'ignored suggestion', 'weight': 0.25},
            {'text': 'backhanded compliment', 'weight': 0.3},
        ]
        
        # Select based on relationship dynamics
        aggression = templates[hash((perpetrator, target)) % len(templates)]
        
        return {
            'perpetrator': perpetrator,
            'target': target,
            'action': aggression['text'],
            'weight': aggression['weight'] * context.get('intensity_multiplier', 1.0),
            'context': context
        }

# Integration Functions

def integrate_with_memory_system(accumulator: TensionAccumulator, 
                                memory_store: Dict) -> None:
    """Store tension patterns in memory for future reference"""
    memory_key = f"tension_{accumulator.character_a}_{accumulator.character_b}"
    
    memory_store[memory_key] = {
        'current_pressure': accumulator.current_pressure,
        'pattern': accumulator._get_dominant_tension_source().value,
        'friction_count': len(accumulator.friction_points),
        'last_updated': datetime.now().isoformat(),
        'historical_peaks': memory_store.get(memory_key, {}).get('historical_peaks', []) + 
                           [accumulator.current_pressure] if accumulator.current_pressure > 8 else []
    }

def integrate_with_relationship_system(tension_web: Dict, 
                                      relationships: Dict) -> None:
    """Modify relationship dimensions based on tension"""
    for (char_a, char_b), tension_data in tension_web.items():
        rel_key = f"{char_a}_{char_b}"
        
        if rel_key in relationships:
            # High tension affects trust and comfort
            if tension_data['pressure'] > 8:
                relationships[rel_key]['trust'] *= 0.95
                relationships[rel_key]['comfort'] *= 0.9
                
            # Resolved tensions can increase intimacy
            if tension_data.get('resolved'):
                relationships[rel_key]['intimacy'] *= 1.05

# Agent Functions for LLM Integration

def detect_brewing_tension(scene: Dict, analyzer: TensionAnalyzer) -> str:
    """LLM tool function to detect brewing tensions"""
    opportunities = analyzer.detect_friction_opportunity(scene)
    
    if opportunities:
        return json.dumps({
            'tension_detected': True,
            'opportunities': opportunities,
            'recommended_action': 'Consider introducing subtle friction',
            'intensity': max(opp['intensity'] for opp in opportunities)
        })
    
    return json.dumps({
        'tension_detected': False,
        'message': 'No significant tension opportunities in current scene'
    })

def trigger_tension_event(char_a: str, char_b: str, 
                         analyzer: TensionAnalyzer) -> str:
    """LLM tool to check if tension should trigger an event"""
    key = tuple(sorted([char_a, char_b]))
    
    if key in analyzer.accumulators:
        accumulator = analyzer.accumulators[key]
        emergence = accumulator.check_boiling_point()
        
        if emergence:
            return json.dumps({
                'should_trigger': True,
                'event_data': emergence,
                'suggestion': f"Create {emergence['intensity_level']} level conflict about {emergence['trigger_suggestion']}"
            })
    
    return json.dumps({'should_trigger': False})
