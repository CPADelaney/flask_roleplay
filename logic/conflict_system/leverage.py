# Meaningful Objects and Social Leverage Systems

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

# MEANINGFUL OBJECTS SYSTEM

class ObjectSignificance(Enum):
    """Types of significance an object can hold"""
    POWER_SYMBOL = "power_symbol"           # Represents dominance/submission
    COMFORT_ITEM = "comfort_item"           # Emotional security
    BOUNDARY_MARKER = "boundary_marker"     # Territorial claim
    GIFT_BOND = "gift_bond"                # Relationship token
    ROUTINE_ANCHOR = "routine_anchor"       # Part of daily ritual
    STATUS_DISPLAY = "status_display"       # Social hierarchy marker
    MEMORY_VESSEL = "memory_vessel"         # Holds shared history
    CONTROL_TOOL = "control_tool"          # Enables control patterns

@dataclass
class MeaningfulObject:
    """An object with emotional/power significance"""
    name: str
    owner: Optional[str]
    significance_type: ObjectSignificance
    power_weight: float  # -1 to 1, negative = submission, positive = dominance
    emotional_charge: float  # 0 to 1, how much it matters
    associated_characters: Set[str] = field(default_factory=set)
    precedents: List[Dict] = field(default_factory=list)
    location: str = "unknown"
    is_contested: bool = False
    creation_context: Dict = field(default_factory=dict)
    
    def add_precedent(self, event: Dict):
        """Record a precedent involving this object"""
        self.precedents.append({
            'timestamp': datetime.now(),
            'event': event,
            'participants': event.get('participants', []),
            'outcome': event.get('outcome'),
            'power_shift': event.get('power_shift', 0)
        })
        
        # Adjust power weight based on precedent
        if event.get('power_shift'):
            self.power_weight = min(1, max(-1, 
                self.power_weight + event['power_shift'] * 0.1))
    
    def calculate_leverage_value(self, for_character: str, 
                                 against_character: str) -> float:
        """Calculate how much leverage this object provides"""
        leverage = 0.0
        
        # Ownership provides base leverage
        if self.owner == for_character:
            leverage += 0.3
            
        # Emotional charge multiplies leverage
        leverage *= (1 + self.emotional_charge)
        
        # Check precedents for specific character interactions
        for precedent in self.precedents:
            if against_character in precedent['participants']:
                if precedent['outcome'] == 'submission':
                    leverage += 0.2
                elif precedent['outcome'] == 'resistance':
                    leverage -= 0.1
                    
        return min(1.0, leverage)

class ObjectGenerator:
    """Generates meaningful objects from conflicts and events"""
    
    def __init__(self):
        self.object_templates = {
            'domestic': [
                {'name': 'favorite mug', 'significance': ObjectSignificance.ROUTINE_ANCHOR},
                {'name': 'special chair', 'significance': ObjectSignificance.BOUNDARY_MARKER},
                {'name': 'gifted jewelry', 'significance': ObjectSignificance.GIFT_BOND},
                {'name': 'household keys', 'significance': ObjectSignificance.CONTROL_TOOL},
                {'name': 'photo album', 'significance': ObjectSignificance.MEMORY_VESSEL},
                {'name': 'personal diary', 'significance': ObjectSignificance.BOUNDARY_MARKER},
            ],
            'power': [
                {'name': 'collar', 'significance': ObjectSignificance.POWER_SYMBOL},
                {'name': 'special privileges card', 'significance': ObjectSignificance.STATUS_DISPLAY},
                {'name': 'punishment implement', 'significance': ObjectSignificance.CONTROL_TOOL},
                {'name': 'reward token', 'significance': ObjectSignificance.GIFT_BOND},
            ]
        }
    
    def generate_from_conflict(self, conflict: Dict) -> Optional[MeaningfulObject]:
        """Create object that emerges from conflict resolution"""
        if conflict['resolution_type'] == 'established_pattern':
            # Pattern-based objects become routine anchors
            return MeaningfulObject(
                name=f"{conflict['winner']}'s {self._generate_object_name(conflict)}",
                owner=conflict['winner'],
                significance_type=ObjectSignificance.ROUTINE_ANCHOR,
                power_weight=0.3 if conflict['winner'] else -0.3,
                emotional_charge=conflict.get('intensity', 0.5),
                associated_characters=set(conflict['participants']),
                creation_context=conflict
            )
        elif conflict['resolution_type'] == 'gift_reconciliation':
            # Gifts become relationship tokens
            return MeaningfulObject(
                name=self._generate_gift_name(conflict),
                owner=conflict['recipient'],
                significance_type=ObjectSignificance.GIFT_BOND,
                power_weight=0.0,  # Neutral, but can shift
                emotional_charge=0.7,
                associated_characters=set(conflict['participants']),
                creation_context=conflict
            )
        return None
    
    def _generate_object_name(self, context: Dict) -> str:
        """Generate contextual object name"""
        object_types = ['chair', 'mug', 'spot', 'time slot', 'privilege', 'rule']
        return object_types[hash(str(context)) % len(object_types)]
    
    def _generate_gift_name(self, context: Dict) -> str:
        """Generate appropriate gift name"""
        gifts = ['bracelet', 'book', 'special meal', 'handwritten note', 
                 'small trinket', 'flower', 'custom playlist']
        return gifts[hash(str(context)) % len(gifts)]

# SOCIAL LEVERAGE SYSTEM

class LeverageType(Enum):
    """Types of social leverage"""
    EMOTIONAL_DEBT = "emotional_debt"       # Guilt, obligation
    PATTERN_PRECEDENT = "pattern_precedent" # "You always..."
    SECRET_KNOWLEDGE = "secret_knowledge"   # Private information
    SOCIAL_CAPITAL = "social_capital"       # Reputation, standing
    RECIPROCITY = "reciprocity"            # Favor economy
    DEPENDENCY = "dependency"               # Need-based
    AUTHORITY = "authority"                # Hierarchical power
    INTIMACY = "intimacy"                  # Emotional closeness

@dataclass
class SocialLeverage:
    """Represents leverage one character has over another"""
    holder: str
    target: str
    leverage_type: LeverageType
    strength: float  # 0 to 1
    context: Dict
    expiration: Optional[datetime] = None
    uses_remaining: int = -1  # -1 for unlimited
    evidence: List[Dict] = field(default_factory=list)
    
    def use(self, context: Dict) -> Tuple[bool, float]:
        """Attempt to use leverage"""
        if self.expiration and datetime.now() > self.expiration:
            return False, 0.0
            
        if self.uses_remaining == 0:
            return False, 0.0
            
        # Calculate effectiveness based on context
        effectiveness = self.strength
        
        # Reduce effectiveness with repeated use
        if self.uses_remaining > 0:
            self.uses_remaining -= 1
            effectiveness *= (self.uses_remaining / max(1, self.uses_remaining + 1))
            
        # Context modifiers
        if context.get('public_setting') and self.leverage_type == LeverageType.SECRET_KNOWLEDGE:
            effectiveness *= 1.5  # Secrets more powerful in public
        elif context.get('private_setting') and self.leverage_type == LeverageType.INTIMACY:
            effectiveness *= 1.3  # Intimacy more effective in private
            
        # Record use
        self.evidence.append({
            'timestamp': datetime.now(),
            'context': context,
            'effectiveness': effectiveness
        })
        
        return True, effectiveness
    
    def decay(self, days_passed: int):
        """Natural decay of leverage over time"""
        decay_rates = {
            LeverageType.EMOTIONAL_DEBT: 0.05,
            LeverageType.PATTERN_PRECEDENT: 0.02,  # Patterns decay slowly
            LeverageType.SECRET_KNOWLEDGE: 0.01,   # Secrets keep power
            LeverageType.SOCIAL_CAPITAL: 0.08,
            LeverageType.RECIPROCITY: 0.10,        # Favors fade fast
            LeverageType.DEPENDENCY: 0.03,
            LeverageType.AUTHORITY: 0.02,
            LeverageType.INTIMACY: 0.04
        }
        
        decay = decay_rates.get(self.leverage_type, 0.05)
        self.strength *= (1 - decay * days_passed)
        self.strength = max(0, self.strength)

class LeverageSystem:
    """Manages social leverage between characters"""
    
    def __init__(self):
        self.leverage_map: Dict[Tuple[str, str], List[SocialLeverage]] = {}
        self.favor_ledger: Dict[Tuple[str, str], float] = {}  # Track favor balance
        self.patterns_db: Dict[str, List[Dict]] = {}  # Character behavior patterns
        
    def create_leverage(self, holder: str, target: str, 
                       leverage_type: LeverageType, context: Dict) -> SocialLeverage:
        """Create new leverage based on events"""
        
        # Calculate initial strength based on type and context
        base_strength = {
            LeverageType.EMOTIONAL_DEBT: 0.6,
            LeverageType.PATTERN_PRECEDENT: 0.4,
            LeverageType.SECRET_KNOWLEDGE: 0.8,
            LeverageType.SOCIAL_CAPITAL: 0.5,
            LeverageType.RECIPROCITY: 0.5,
            LeverageType.DEPENDENCY: 0.7,
            LeverageType.AUTHORITY: 0.6,
            LeverageType.INTIMACY: 0.5
        }
        
        leverage = SocialLeverage(
            holder=holder,
            target=target,
            leverage_type=leverage_type,
            strength=base_strength.get(leverage_type, 0.5),
            context=context
        )
        
        # Set expiration for temporary leverage
        if leverage_type == LeverageType.RECIPROCITY:
            leverage.expiration = datetime.now() + timedelta(days=7)
            leverage.uses_remaining = 1
        elif leverage_type == LeverageType.EMOTIONAL_DEBT:
            leverage.uses_remaining = 3  # Can guilt trip a few times
            
        # Store leverage
        key = (holder, target)
        if key not in self.leverage_map:
            self.leverage_map[key] = []
        self.leverage_map[key].append(leverage)
        
        return leverage
    
    def detect_pattern_leverage(self, character: str, 
                               behavior_history: List[Dict]) -> List[Dict]:
        """Detect patterns that could become leverage"""
        patterns = []
        
        # Look for repeated behaviors
        behavior_counts = {}
        for behavior in behavior_history[-20:]:  # Last 20 behaviors
            key = behavior['type']
            behavior_counts[key] = behavior_counts.get(key, 0) + 1
            
        # Patterns that appear 3+ times become leverage
        for behavior_type, count in behavior_counts.items():
            if count >= 3:
                patterns.append({
                    'character': character,
                    'pattern': behavior_type,
                    'frequency': count,
                    'leverage_potential': min(1.0, count * 0.15),
                    'example_phrase': f"You always {behavior_type}"
                })
                
        return patterns
    
    def calculate_total_leverage(self, holder: str, target: str) -> float:
        """Calculate total leverage one character has over another"""
        key = (holder, target)
        if key not in self.leverage_map:
            return 0.0
            
        total = 0.0
        for leverage in self.leverage_map[key]:
            if leverage.strength > 0:
                total += leverage.strength
                
        # Check favor balance
        favor_key = (holder, target)
        if favor_key in self.favor_ledger:
            total += self.favor_ledger[favor_key] * 0.1
            
        return min(1.0, total)
    
    def record_favor(self, giver: str, receiver: str, value: float):
        """Record a favor in the ledger"""
        key = (receiver, giver)  # Receiver owes giver
        self.favor_ledger[key] = self.favor_ledger.get(key, 0) + value
        
    def cash_in_favor(self, collector: str, debtor: str) -> Tuple[bool, float]:
        """Attempt to cash in a favor"""
        key = (debtor, collector)
        
        if key in self.favor_ledger and self.favor_ledger[key] > 0:
            value = min(1.0, self.favor_ledger[key])
            self.favor_ledger[key] = max(0, self.favor_ledger[key] - 1.0)
            return True, value
            
        return False, 0.0

# Integration Functions

def integrate_objects_with_conflicts(obj: MeaningfulObject, 
                                    conflict: Dict) -> Dict:
    """Use objects as stakes or tools in conflicts"""
    integration = {
        'object': obj.name,
        'role': None,
        'impact': 0.0
    }
    
    if obj.is_contested:
        integration['role'] = 'stake'
        integration['impact'] = obj.emotional_charge
    elif obj.owner in conflict['participants']:
        integration['role'] = 'leverage_tool'
        integration['impact'] = obj.power_weight
    else:
        integration['role'] = 'catalyst'
        integration['impact'] = 0.3
        
    return integration

def discover_hidden_pattern(character: str, leverage_system: LeverageSystem,
                          behavior_history: List[Dict]) -> str:
    """LLM tool to discover manipulation patterns"""
    patterns = leverage_system.detect_pattern_leverage(character, behavior_history)
    
    if patterns:
        most_significant = max(patterns, key=lambda p: p['leverage_potential'])
        return json.dumps({
            'pattern_found': True,
            'character': character,
            'pattern': most_significant['pattern'],
            'frequency': most_significant['frequency'],
            'leverage_created': True,
            'suggested_dialogue': most_significant['example_phrase']
        })
    
    return json.dumps({
        'pattern_found': False,
        'character': character,
        'message': 'No significant patterns detected yet'
    })

def calculate_social_leverage(char_a: str, char_b: str, 
                             leverage_system: LeverageSystem) -> str:
    """LLM tool to check leverage dynamics"""
    leverage_a_to_b = leverage_system.calculate_total_leverage(char_a, char_b)
    leverage_b_to_a = leverage_system.calculate_total_leverage(char_b, char_a)
    
    balance = leverage_a_to_b - leverage_b_to_a
    
    return json.dumps({
        'leverage_a_to_b': leverage_a_to_b,
        'leverage_b_to_a': leverage_b_to_a,
        'balance': balance,
        'dominant': char_a if balance > 0 else char_b if balance < 0 else 'balanced',
        'recommendation': _suggest_leverage_use(balance)
    })

def _suggest_leverage_use(balance: float) -> str:
    """Suggest how to use leverage based on balance"""
    if abs(balance) < 0.2:
        return "Relatively balanced - small favors or requests appropriate"
    elif balance > 0.5:
        return "Strong leverage position - can make significant requests"
    elif balance < -0.5:
        return "Weak position - focus on building leverage or offering favors"
    else:
        return "Moderate leverage - standard requests reasonable"

# Database Schema SQL
SCHEMA_SQL = """
-- Meaningful Objects Table
CREATE TABLE meaningful_objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    owner TEXT,
    significance_type TEXT NOT NULL,
    power_weight REAL DEFAULT 0.0,
    emotional_charge REAL DEFAULT 0.5,
    location TEXT DEFAULT 'unknown',
    is_contested BOOLEAN DEFAULT FALSE,
    creation_context JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Object Precedents Table
CREATE TABLE object_precedents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id INTEGER REFERENCES meaningful_objects(id),
    event JSON NOT NULL,
    participants JSON,
    outcome TEXT,
    power_shift REAL DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Social Leverage Table
CREATE TABLE social_leverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    holder TEXT NOT NULL,
    target TEXT NOT NULL,
    leverage_type TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    context JSON,
    expiration TIMESTAMP,
    uses_remaining INTEGER DEFAULT -1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Favor Ledger Table
CREATE TABLE favor_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    giver TEXT NOT NULL,
    receiver TEXT NOT NULL,
    favor_value REAL DEFAULT 1.0,
    description TEXT,
    repaid BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pattern Evidence Table
CREATE TABLE pattern_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    evidence JSON,
    frequency INTEGER DEFAULT 1,
    last_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
