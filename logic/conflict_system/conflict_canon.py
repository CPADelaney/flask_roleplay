# Canon System & Investigation/Discovery Mechanics

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

# CANON SYSTEM - Making conflict outcomes permanent facts

class CanonType(Enum):
    """Types of canonical facts that can be established"""
    SOCIAL_HIERARCHY = "social_hierarchy"      # Who defers to whom
    ROUTINE_OWNERSHIP = "routine_ownership"    # Who owns what time/space
    BEHAVIORAL_PATTERN = "behavioral_pattern"  # Established character patterns
    RELATIONSHIP_FACT = "relationship_fact"    # Accepted relationship dynamics
    HOUSE_RULE = "house_rule"                 # Established household rules
    PRECEDENT = "precedent"                   # Past events that set standards
    PREFERENCE = "preference"                  # Accepted personal preferences
    BOUNDARY = "boundary"                     # Established personal boundaries

@dataclass
class CanonicalFact:
    """A fact that has become accepted truth in the game world"""
    fact_id: str
    canon_type: CanonType
    statement: str  # Human-readable fact
    subjects: Set[str]  # Characters involved
    established_date: datetime
    strength: float = 1.0  # How strongly established (0-1)
    challenges: int = 0  # Times it's been challenged
    last_referenced: datetime = field(default_factory=datetime.now)
    evidence: List[Dict] = field(default_factory=list)
    modifiable: bool = True  # Can be changed through conflict
    
    def reinforce(self, context: Dict):
        """Strengthen fact through reference or use"""
        self.strength = min(1.0, self.strength + 0.1)
        self.last_referenced = datetime.now()
        self.evidence.append({
            'type': 'reinforcement',
            'context': context,
            'timestamp': datetime.now()
        })
    
    def challenge(self, challenger: str, success: bool):
        """Record a challenge to this fact"""
        self.challenges += 1
        if success:
            self.strength *= 0.7  # Successful challenges weaken facts
            if self.strength < 0.3:
                self.modifiable = True  # Becomes changeable
        else:
            self.strength = min(1.0, self.strength + 0.05)  # Failed challenges strengthen
        
        self.evidence.append({
            'type': 'challenge',
            'challenger': challenger,
            'success': success,
            'timestamp': datetime.now()
        })
    
    def to_narrative(self) -> str:
        """Convert to narrative description"""
        narratives = {
            CanonType.SOCIAL_HIERARCHY: f"Everyone accepts that {self.statement}",
            CanonType.ROUTINE_OWNERSHIP: f"It's established that {self.statement}",
            CanonType.BEHAVIORAL_PATTERN: f"Everyone knows that {self.statement}",
            CanonType.RELATIONSHIP_FACT: f"It's understood that {self.statement}",
            CanonType.HOUSE_RULE: f"The house rule is: {self.statement}",
            CanonType.PRECEDENT: f"As established before, {self.statement}",
            CanonType.PREFERENCE: f"It's accepted that {self.statement}",
            CanonType.BOUNDARY: f"The boundary is clear: {self.statement}"
        }
        return narratives.get(self.canon_type, self.statement)

class CanonSystem:
    """Manages canonical facts and social truths"""
    
    def __init__(self):
        self.facts: Dict[str, CanonicalFact] = {}
        self.fact_index: Dict[str, Set[str]] = {}  # Character -> fact IDs
        self.pending_facts: List[Dict] = []  # Facts waiting to be canonized
        
    def establish_fact(self, canon_type: CanonType, statement: str, 
                      subjects: Set[str], evidence: List[Dict]) -> CanonicalFact:
        """Establish a new canonical fact"""
        
        # Generate unique ID
        fact_id = self._generate_fact_id(statement, subjects)
        
        # Check if similar fact exists
        existing = self._find_similar_fact(statement, subjects)
        if existing:
            existing.reinforce({'new_evidence': evidence})
            return existing
        
        # Create new fact
        fact = CanonicalFact(
            fact_id=fact_id,
            canon_type=canon_type,
            statement=statement,
            subjects=subjects,
            established_date=datetime.now(),
            evidence=evidence
        )
        
        # Store and index
        self.facts[fact_id] = fact
        for subject in subjects:
            if subject not in self.fact_index:
                self.fact_index[subject] = set()
            self.fact_index[subject].add(fact_id)
        
        return fact
    
    def _generate_fact_id(self, statement: str, subjects: Set[str]) -> str:
        """Generate unique ID for fact"""
        content = f"{statement}_{sorted(subjects)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _find_similar_fact(self, statement: str, subjects: Set[str]) -> Optional[CanonicalFact]:
        """Find existing similar fact"""
        for fact in self.facts.values():
            if fact.subjects == subjects:
                # Check similarity (simple word overlap for now)
                overlap = len(set(statement.split()) & set(fact.statement.split()))
                if overlap > len(statement.split()) * 0.5:
                    return fact
        return None
    
    def get_character_facts(self, character: str) -> List[CanonicalFact]:
        """Get all canonical facts involving a character"""
        fact_ids = self.fact_index.get(character, set())
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]
    
    def check_fact_conflict(self, proposed_action: Dict) -> List[CanonicalFact]:
        """Check if an action conflicts with established facts"""
        conflicts = []
        
        for fact in self.facts.values():
            # Check if action violates fact
            if self._action_violates_fact(proposed_action, fact):
                conflicts.append(fact)
        
        return conflicts
    
    def _action_violates_fact(self, action: Dict, fact: CanonicalFact) -> bool:
        """Check if action violates established fact"""
        # Check routine violations
        if fact.canon_type == CanonType.ROUTINE_OWNERSHIP:
            if action.get('type') == 'use_routine' and action.get('time') in fact.statement:
                return action.get('actor') not in fact.subjects
        
        # Check boundary violations
        if fact.canon_type == CanonType.BOUNDARY:
            if action.get('target') in fact.subjects:
                return action.get('type') in ['intrude', 'override', 'ignore_boundary']
        
        return False
    
    def evolve_fact(self, fact_id: str, new_statement: str, 
                   trigger_event: Dict) -> CanonicalFact:
        """Evolve an existing fact based on events"""
        if fact_id not in self.facts:
            return None
        
        old_fact = self.facts[fact_id]
        
        # Create evolved fact
        new_fact = CanonicalFact(
            fact_id=f"{fact_id}_v2",
            canon_type=old_fact.canon_type,
            statement=new_statement,
            subjects=old_fact.subjects,
            established_date=datetime.now(),
            evidence=[{
                'type': 'evolution',
                'from_fact': fact_id,
                'trigger': trigger_event
            }] + old_fact.evidence[-3:]  # Keep some history
        )
        
        # Replace old with new
        self.facts[new_fact.fact_id] = new_fact
        del self.facts[fact_id]
        
        # Update index
        for subject in new_fact.subjects:
            self.fact_index[subject].discard(fact_id)
            self.fact_index[subject].add(new_fact.fact_id)
        
        return new_fact

# INVESTIGATION & DISCOVERY MECHANICS

class DiscoveryType(Enum):
    """Types of discoveries player can make"""
    HIDDEN_PATTERN = "hidden_pattern"          # Behavioral patterns
    MANIPULATION = "manipulation"              # Someone manipulating
    SECRET_ALLIANCE = "secret_alliance"        # Hidden cooperation
    POWER_DYNAMIC = "power_dynamic"           # Unspoken hierarchy
    EMOTIONAL_STATE = "emotional_state"       # Hidden feelings
    PAST_EVENT = "past_event"                # Unknown history
    TRUE_PREFERENCE = "true_preference"       # Real vs stated preference
    LEVERAGE_POINT = "leverage_point"         # Potential leverage

@dataclass
class Investigation:
    """An ongoing investigation by the player"""
    investigation_id: str
    target: str  # What/who is being investigated
    discovery_type: DiscoveryType
    clues_found: List[Dict] = field(default_factory=list)
    progress: float = 0.0  # 0-1, 1 = discovery made
    started: datetime = field(default_factory=datetime.now)
    completed: bool = False
    discovery: Optional[Dict] = None
    methods_used: Set[str] = field(default_factory=set)
    
    def add_clue(self, clue: Dict, value: float = 0.2):
        """Add a clue to the investigation"""
        self.clues_found.append({
            **clue,
            'timestamp': datetime.now(),
            'value': value
        })
        self.progress = min(1.0, self.progress + value)
        
        # Check for discovery
        if self.progress >= 1.0 and not self.completed:
            self.completed = True
            self.discovery = self._synthesize_discovery()
    
    def _synthesize_discovery(self) -> Dict:
        """Synthesize clues into a discovery"""
        if self.discovery_type == DiscoveryType.HIDDEN_PATTERN:
            return {
                'type': 'pattern_revealed',
                'target': self.target,
                'pattern': self._extract_pattern(),
                'evidence': self.clues_found,
                'confidence': min(1.0, len(self.clues_found) * 0.15)
            }
        elif self.discovery_type == DiscoveryType.MANIPULATION:
            return {
                'type': 'manipulation_uncovered',
                'manipulator': self.target,
                'tactics': [c['tactic'] for c in self.clues_found if 'tactic' in c],
                'victims': set(c.get('victim') for c in self.clues_found if 'victim' in c)
            }
        # Add more discovery synthesis types...
        return {'type': 'generic_discovery', 'clues': self.clues_found}
    
    def _extract_pattern(self) -> str:
        """Extract pattern from clues"""
        behaviors = [c.get('behavior', '') for c in self.clues_found]
        # Find most common behavior
        if behaviors:
            return max(set(behaviors), key=behaviors.count)
        return "unknown pattern"

class InvestigationSystem:
    """Manages investigations and discoveries"""
    
    def __init__(self):
        self.active_investigations: Dict[str, Investigation] = {}
        self.completed_investigations: List[Investigation] = []
        self.discovery_triggers: Dict[str, List[Dict]] = {}  # What triggers investigations
        self.clue_sources: Dict[str, List[str]] = {
            'observation': ['watch_carefully', 'note_behavior', 'track_patterns'],
            'conversation': ['ask_others', 'casual_questions', 'deep_talk'],
            'snooping': ['check_phone', 'read_diary', 'search_room'],
            'experimentation': ['test_reaction', 'create_situation', 'provoke'],
            'deduction': ['connect_dots', 'analyze_patterns', 'timeline']
        }
    
    def start_investigation(self, target: str, discovery_type: DiscoveryType,
                          initial_clue: Optional[Dict] = None) -> Investigation:
        """Start a new investigation"""
        investigation_id = f"inv_{target}_{discovery_type.value}_{datetime.now().timestamp()}"
        
        investigation = Investigation(
            investigation_id=investigation_id,
            target=target,
            discovery_type=discovery_type
        )
        
        if initial_clue:
            investigation.add_clue(initial_clue, value=0.3)
        
        self.active_investigations[investigation_id] = investigation
        return investigation
    
    def add_clue_opportunity(self, scene_context: Dict) -> List[Dict]:
        """Generate clue opportunities based on scene"""
        opportunities = []
        
        for inv_id, investigation in self.active_investigations.items():
            if not investigation.completed:
                # Check if target is in scene
                if investigation.target in scene_context.get('present_characters', []):
                    opp = self._generate_clue_opportunity(investigation, scene_context)
                    if opp:
                        opportunities.append(opp)
        
        return opportunities
    
    def _generate_clue_opportunity(self, investigation: Investigation, 
                                  context: Dict) -> Optional[Dict]:
        """Generate a specific clue opportunity"""
        
        if investigation.discovery_type == DiscoveryType.HIDDEN_PATTERN:
            # Observation opportunity
            return {
                'investigation_id': investigation.investigation_id,
                'method': 'observation',
                'action': 'Watch how they interact with others',
                'potential_clue': {
                    'behavior': 'deflection_pattern',
                    'context': context.get('activity', 'conversation')
                },
                'difficulty': 0.3,  # How hard to notice
                'time_cost': 'low'
            }
        
        elif investigation.discovery_type == DiscoveryType.MANIPULATION:
            # Conversation opportunity
            if 'victim' in context:
                return {
                    'investigation_id': investigation.investigation_id,
                    'method': 'conversation',
                    'action': f"Casually ask {context['victim']} about {investigation.target}",
                    'potential_clue': {
                        'tactic': 'guilt_trips',
                        'victim': context['victim'],
                        'frequency': 'often'
                    },
                    'difficulty': 0.5,
                    'time_cost': 'medium'
                }
        
        elif investigation.discovery_type == DiscoveryType.SECRET_ALLIANCE:
            # Snooping opportunity
            return {
                'investigation_id': investigation.investigation_id,
                'method': 'snooping',
                'action': 'Check their recent messages',
                'potential_clue': {
                    'communication': 'frequent',
                    'topic': 'coordination',
                    'ally': 'unknown'
                },
                'difficulty': 0.7,
                'time_cost': 'high',
                'risk': 'might_get_caught'
            }
        
        return None
    
    def gather_clue(self, investigation_id: str, method: str, 
                   success: bool, clue_data: Dict) -> Dict:
        """Process clue gathering attempt"""
        if investigation_id not in self.active_investigations:
            return {'error': 'Investigation not found'}
        
        investigation = self.active_investigations[investigation_id]
        investigation.methods_used.add(method)
        
        if success:
            # Adjust clue value based on method
            value_modifiers = {
                'observation': 0.15,
                'conversation': 0.20,
                'snooping': 0.35,
                'experimentation': 0.25,
                'deduction': 0.30
            }
            
            base_value = value_modifiers.get(method, 0.2)
            
            # Add randomness
            import random
            actual_value = base_value * (0.8 + random.random() * 0.4)
            
            investigation.add_clue(clue_data, actual_value)
            
            result = {
                'success': True,
                'progress': investigation.progress,
                'clue': clue_data,
                'message': self._generate_clue_message(clue_data)
            }
            
            # Check for discovery
            if investigation.completed:
                result['discovery_made'] = True
                result['discovery'] = investigation.discovery
                
                # Move to completed
                self.completed_investigations.append(investigation)
                del self.active_investigations[investigation_id]
            
            return result
        else:
            return {
                'success': False,
                'message': 'No useful information gathered',
                'progress': investigation.progress
            }
    
    def _generate_clue_message(self, clue: Dict) -> str:
        """Generate narrative message for clue discovery"""
        messages = {
            'deflection_pattern': "You notice they always change the subject when...",
            'guilt_trips': "They seem to use guilt as a tool frequently",
            'coordination': "There's definitely some coordination happening",
            'frequent': "They communicate more than you realized"
        }
        
        for key, msg in messages.items():
            if key in str(clue.values()):
                return msg
        
        return "You learned something interesting"
    
    def reveal_discovery(self, discovery: Dict) -> Dict:
        """Process and reveal a completed discovery"""
        revelation = {
            'type': discovery['type'],
            'impact': self._calculate_discovery_impact(discovery),
            'new_options': self._generate_new_options(discovery),
            'narrative': self._generate_revelation_narrative(discovery)
        }
        
        return revelation
    
    def _calculate_discovery_impact(self, discovery: Dict) -> Dict:
        """Calculate impact of discovery on game state"""
        impact = {}
        
        if discovery['type'] == 'manipulation_uncovered':
            impact['leverage_gained'] = True
            impact['relationship_strain'] = -0.2
            impact['trust_damaged'] = True
            
        elif discovery['type'] == 'pattern_revealed':
            impact['prediction_possible'] = True
            impact['counter_strategies'] = ['preempt', 'confront', 'avoid']
            
        return impact
    
    def _generate_new_options(self, discovery: Dict) -> List[str]:
        """Generate new player options from discovery"""
        options = []
        
        if discovery['type'] == 'manipulation_uncovered':
            options.extend([
                'Confront the manipulator',
                'Warn the victims',
                'Use knowledge as leverage',
                'Gather more evidence'
            ])
            
        elif discovery['type'] == 'pattern_revealed':
            options.extend([
                'Exploit the pattern',
                'Break the pattern',
                'Point it out to them',
                'Use it to predict behavior'
            ])
            
        return options
    
    def _generate_revelation_narrative(self, discovery: Dict) -> str:
        """Generate narrative description of discovery"""
        if discovery['type'] == 'manipulation_uncovered':
            return f"You've discovered that {discovery['manipulator']} has been manipulating {', '.join(discovery['victims'])} using {', '.join(discovery['tactics'])}"
        
        elif discovery['type'] == 'pattern_revealed':
            return f"You've identified a clear pattern: {discovery['pattern']}"
        
        return "You've made an important discovery"

# Integration with Conflict System

def integrate_canon_with_conflicts(conflict: Dict, resolution: Dict, 
                                  canon_system: CanonSystem) -> CanonicalFact:
    """Create canonical facts from conflict resolutions"""
    
    # Determine fact type based on conflict
    if conflict['type'] == 'routine_dispute':
        canon_type = CanonType.ROUTINE_OWNERSHIP
        statement = f"{resolution['winner']} owns the {conflict['disputed_routine']}"
    elif conflict['type'] == 'boundary_setting':
        canon_type = CanonType.BOUNDARY  
        statement = resolution['established_boundary']
    elif conflict['type'] == 'hierarchy_challenge':
        canon_type = CanonType.SOCIAL_HIERARCHY
        statement = f"{resolution['dominant']} has authority over {resolution['submissive']}"
    else:
        canon_type = CanonType.PRECEDENT
        statement = resolution.get('outcome', 'Resolution reached')
    
    # Create canonical fact
    fact = canon_system.establish_fact(
        canon_type=canon_type,
        statement=statement,
        subjects=set(conflict['participants']),
        evidence=[{
            'conflict': conflict,
            'resolution': resolution,
            'timestamp': datetime.now()
        }]
    )
    
    return fact

def detect_investigation_triggers(event: Dict, npc_state: Dict,
                                 investigation_system: InvestigationSystem) -> Optional[Investigation]:
    """Detect when player might want to investigate something"""
    
    triggers = []
    
    # Suspicious behavior
    if event.get('type') == 'unexpected_behavior':
        triggers.append({
            'target': event['actor'],
            'type': DiscoveryType.HIDDEN_PATTERN,
            'reason': 'Acting out of character'
        })
    
    # Repeated deflection
    if event.get('deflection_count', 0) > 3:
        triggers.append({
            'target': event['actor'],
            'type': DiscoveryType.TRUE_PREFERENCE,
            'reason': 'Avoiding something'
        })
    
    # Alliance indicators
    if event.get('coordinated_action'):
        triggers.append({
            'target': event['actors'][0],
            'type': DiscoveryType.SECRET_ALLIANCE,
            'reason': 'Suspicious coordination'
        })
    
    # Start investigation if triggered
    if triggers:
        trigger = triggers[0]  # Take first trigger
        return investigation_system.start_investigation(
            target=trigger['target'],
            discovery_type=trigger['type'],
            initial_clue={'trigger_reason': trigger['reason']}
        )
    
    return None

# Database Schema Extensions

CANON_INVESTIGATION_SCHEMA = """
-- Canonical Facts Table
CREATE TABLE canonical_facts (
    fact_id TEXT PRIMARY KEY,
    canon_type TEXT NOT NULL,
    statement TEXT NOT NULL,
    subjects JSON NOT NULL,
    established_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strength REAL DEFAULT 1.0,
    challenges INTEGER DEFAULT 0,
    last_referenced TIMESTAMP,
    evidence JSON,
    modifiable BOOLEAN DEFAULT TRUE
);

-- Fact References Table (track when facts are invoked)
CREATE TABLE fact_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT REFERENCES canonical_facts(fact_id),
    context JSON,
    reference_type TEXT,  -- 'reinforce', 'challenge', 'mention'
    success BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Investigations Table
CREATE TABLE investigations (
    investigation_id TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    discovery_type TEXT NOT NULL,
    progress REAL DEFAULT 0.0,
    started TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed BOOLEAN DEFAULT FALSE,
    discovery JSON,
    methods_used JSON
);

-- Investigation Clues Table
CREATE TABLE investigation_clues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT REFERENCES investigations(investigation_id),
    clue_data JSON NOT NULL,
    value REAL DEFAULT 0.2,
    method TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Discovery Log Table
CREATE TABLE discoveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discovery_type TEXT NOT NULL,
    discovery_data JSON NOT NULL,
    impact JSON,
    revealed_to JSON,  -- Which characters know
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Example LLM Tool Functions

def check_canonical_facts(character: str, action: str, 
                         canon_system: CanonSystem) -> str:
    """LLM tool to check if action violates canonical facts"""
    proposed_action = {
        'actor': character,
        'type': action,
        'timestamp': datetime.now()
    }
    
    conflicts = canon_system.check_fact_conflict(proposed_action)
    
    if conflicts:
        return json.dumps({
            'conflicts_found': True,
            'violations': [
                {
                    'fact': fact.statement,
                    'type': fact.canon_type.value,
                    'strength': fact.strength,
                    'narrative': fact.to_narrative()
                }
                for fact in conflicts
            ],
            'recommendation': 'Action would violate established facts - consider alternative or challenge the fact'
        })
    
    return json.dumps({
        'conflicts_found': False,
        'message': 'Action does not violate any established facts'
    })

def get_investigation_status(investigation_system: InvestigationSystem) -> str:
    """LLM tool to check investigation progress"""
    status = {
        'active_investigations': [],
        'available_methods': investigation_system.clue_sources,
        'recent_discoveries': []
    }
    
    for inv_id, investigation in investigation_system.active_investigations.items():
        status['active_investigations'].append({
            'target': investigation.target,
            'type': investigation.discovery_type.value,
            'progress': f"{investigation.progress:.0%}",
            'clues_found': len(investigation.clues_found),
            'methods_used': list(investigation.methods_used)
        })
    
    for discovery in investigation_system.completed_investigations[-3:]:
        if discovery.discovery:
            status['recent_discoveries'].append(discovery.discovery)
    
    return json.dumps(status, indent=2)
