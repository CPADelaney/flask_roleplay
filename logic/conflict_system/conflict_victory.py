# Domestic Power Struggles & Victory Conditions Systems

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from datetime import datetime, timedelta
import json

class PowerDomain(Enum):
    """Areas where domestic power can be exercised"""
    SCHEDULE = "schedule"  # Who decides when things happen
    SPACE = "space"  # Who controls which spaces
    RESOURCES = "resources"  # Who allocates shared resources
    SOCIAL = "social"  # Who makes social decisions
    ROUTINE = "routine"  # Who sets daily patterns
    PREFERENCES = "preferences"  # Whose preferences become defaults
    DECISIONS = "decisions"  # Who has final say

class TakeoverStage(Enum):
    """Stages of gradual control takeover"""
    TESTING = "testing"  # Initial boundary probing
    ESTABLISHING = "establishing"  # Setting precedents
    NORMALIZING = "normalizing"  # Making it routine
    CONSOLIDATING = "consolidating"  # Reinforcing control
    DOMINANT = "dominant"  # Accepted as normal

@dataclass
class PowerStruggle:
    """Represents an ongoing domestic power struggle"""
    struggle_id: str
    domain: PowerDomain
    instigator_id: str
    target_id: str
    current_stage: TakeoverStage
    control_percentage: float  # 0-100, how much control instigator has
    precedents: List[str] = field(default_factory=list)
    resistance_events: List[Dict] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_action: datetime = field(default_factory=datetime.now)
    
class DomesticPowerSystem:
    """Manages gradual power takeovers in domestic settings"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.active_struggles = {}
        self._load_active_struggles()
    
    def initiate_takeover(self, instigator_id: str, target_id: str, domain: PowerDomain, context: Dict) -> PowerStruggle:
        """Start a gradual takeover attempt in a specific domain"""
        struggle_id = f"{instigator_id}_{target_id}_{domain.value}_{datetime.now().timestamp()}"
        
        # Check if there's already an active struggle in this domain
        existing = self._check_existing_struggle(instigator_id, target_id, domain)
        if existing:
            return self._escalate_existing_struggle(existing, context)
        
        struggle = PowerStruggle(
            struggle_id=struggle_id,
            domain=domain,
            instigator_id=instigator_id,
            target_id=target_id,
            current_stage=TakeoverStage.TESTING,
            control_percentage=20.0  # Start with small foothold
        )
        
        # Record initial takeover action
        initial_action = self._generate_initial_action(domain, context)
        struggle.precedents.append(initial_action)
        
        self.active_struggles[struggle_id] = struggle
        self._save_struggle(struggle)
        
        return struggle
    
    def _generate_initial_action(self, domain: PowerDomain, context: Dict) -> str:
        """Generate the first move in a takeover attempt"""
        actions = {
            PowerDomain.SCHEDULE: [
                "Started deciding bedtime without asking",
                "Began setting meal times unilaterally",
                "Started scheduling shared activities alone"
            ],
            PowerDomain.SPACE: [
                "Rearranged shared space without consultation",
                "Claimed the best spot as 'theirs'",
                "Started storing personal items in shared areas"
            ],
            PowerDomain.RESOURCES: [
                "Started managing the household budget",
                "Began rationing shared supplies",
                "Took control of streaming service choices"
            ],
            PowerDomain.ROUTINE: [
                "Established new morning routine for both",
                "Started enforcing quiet hours",
                "Began dictating weekend patterns"
            ]
        }
        
        import random
        return random.choice(actions.get(domain, ["Made initial power move"]))
    
    def advance_takeover(self, struggle_id: str, action: str, success: bool) -> Dict:
        """Progress a takeover attempt based on action outcome"""
        struggle = self.active_struggles.get(struggle_id)
        if not struggle:
            return {"error": "Struggle not found"}
        
        if success:
            # Successful action advances control
            struggle.control_percentage = min(100, struggle.control_percentage + self._calculate_gain(struggle))
            struggle.precedents.append(action)
            
            # Check for stage advancement
            new_stage = self._determine_stage(struggle.control_percentage)
            if new_stage != struggle.current_stage:
                struggle.current_stage = new_stage
                event = {
                    "type": "stage_advancement",
                    "from": struggle.current_stage.value,
                    "to": new_stage.value,
                    "timestamp": datetime.now().isoformat()
                }
                self._trigger_stage_event(struggle, event)
        else:
            # Failed action records resistance
            struggle.resistance_events.append({
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "impact": "blocked"
            })
            struggle.control_percentage = max(0, struggle.control_percentage - 5)
        
        struggle.last_action = datetime.now()
        self._save_struggle(struggle)
        
        return self._generate_status_report(struggle)
    
    def _calculate_gain(self, struggle: PowerStruggle) -> float:
        """Calculate control gain from successful action"""
        base_gain = 10.0
        
        # Modify based on stage
        stage_modifiers = {
            TakeoverStage.TESTING: 1.5,
            TakeoverStage.ESTABLISHING: 1.2,
            TakeoverStage.NORMALIZING: 1.0,
            TakeoverStage.CONSOLIDATING: 0.8,
            TakeoverStage.DOMINANT: 0.5
        }
        
        gain = base_gain * stage_modifiers[struggle.current_stage]
        
        # Reduce gain if many recent resistance events
        recent_resistance = len([r for r in struggle.resistance_events 
                                if datetime.fromisoformat(r['timestamp']) > datetime.now() - timedelta(days=3)])
        if recent_resistance > 2:
            gain *= 0.6
            
        return gain
    
    def _determine_stage(self, control_percentage: float) -> TakeoverStage:
        """Determine takeover stage based on control percentage"""
        if control_percentage < 30:
            return TakeoverStage.TESTING
        elif control_percentage < 50:
            return TakeoverStage.ESTABLISHING
        elif control_percentage < 70:
            return TakeoverStage.NORMALIZING
        elif control_percentage < 90:
            return TakeoverStage.CONSOLIDATING
        else:
            return TakeoverStage.DOMINANT
    
    def execute_coup(self, instigator_id: str, domain: PowerDomain, context: Dict) -> Dict:
        """Attempt a sudden coup in a specific domain"""
        # Coups can only succeed if groundwork has been laid
        existing_control = self._calculate_existing_control(instigator_id, domain)
        
        if existing_control < 40:
            return {
                "success": False,
                "reason": "Insufficient groundwork",
                "message": "The sudden attempt to take control was easily rebuffed"
            }
        
        # Calculate coup success chance
        success_chance = existing_control / 100.0
        success_chance *= self._get_context_modifier(context)
        
        import random
        if random.random() < success_chance:
            # Successful coup
            self._establish_dominance(instigator_id, domain)
            return {
                "success": True,
                "message": f"Successfully took control of {domain.value}",
                "new_reality": self._generate_new_normal(instigator_id, domain)
            }
        else:
            # Failed coup creates backlash
            self._create_backlash(instigator_id, domain)
            return {
                "success": False,
                "reason": "Resistance succeeded",
                "message": "The coup attempt failed and created resentment",
                "consequence": "Lost progress in this domain"
            }
    
    def _establish_dominance(self, controller_id: str, domain: PowerDomain):
        """Establish accepted dominance in a domain"""
        self.db.execute("""
            INSERT OR REPLACE INTO established_dominance
            (controller_id, domain, established_date, acceptance_level)
            VALUES (?, ?, ?, ?)
        """, (controller_id, domain.value, datetime.now(), "accepted"))
        
        # Create lasting precedents
        self._create_permanent_precedents(controller_id, domain)
    
    def establish_default_preference(self, npc_id: str, category: str, preference: str) -> Dict:
        """Set an NPC's preference as the household default"""
        existing_default = self._get_current_default(category)
        
        if existing_default and existing_default['controller_id'] != npc_id:
            # There's already a different default
            return self._challenge_default(npc_id, category, preference, existing_default)
        
        # Establish new default
        self.db.execute("""
            INSERT OR REPLACE INTO default_preferences
            (category, preference, controller_id, established_date, challenge_count)
            VALUES (?, ?, ?, ?, 0)
        """, (category, preference, npc_id, datetime.now()))
        
        return {
            "success": True,
            "message": f"{preference} is now the household default for {category}",
            "impact": "Everyone is expected to accommodate this preference"
        }
    
    def _challenge_default(self, challenger_id: str, category: str, new_preference: str, existing: Dict) -> Dict:
        """Challenge an existing default preference"""
        # Check challenger's power in relevant domain
        challenger_power = self._calculate_domain_power(challenger_id, PowerDomain.PREFERENCES)
        defender_power = self._calculate_domain_power(existing['controller_id'], PowerDomain.PREFERENCES)
        
        if challenger_power > defender_power * 1.2:  # Need significant advantage
            # Successful challenge
            self.db.execute("""
                UPDATE default_preferences
                SET preference = ?, controller_id = ?, established_date = ?
                WHERE category = ?
            """, (new_preference, challenger_id, datetime.now(), category))
            
            return {
                "success": True,
                "message": f"Successfully changed {category} default to {new_preference}",
                "previous": existing['preference']
            }
        else:
            # Failed challenge
            self.db.execute("""
                UPDATE default_preferences
                SET challenge_count = challenge_count + 1
                WHERE category = ?
            """, (category,))
            
            return {
                "success": False,
                "message": f"Failed to change the default from {existing['preference']}",
                "consequence": "The existing default is reinforced"
            }

# Victory Conditions System

class VictoryType(Enum):
    """Types of social victories"""
    NORM_ESTABLISHMENT = "norm_establishment"
    COMFORT_ZONE = "comfort_zone"
    SOCIAL_POSITION = "social_position"
    RELATIONSHIP_DEFINITION = "relationship_definition"
    PRECEDENT_CHAIN = "precedent_chain"
    PYRRHIC = "pyrrhic"  # Won but at great cost

@dataclass
class VictoryCondition:
    """Defines what constitutes a victory in social conflicts"""
    condition_id: str
    victory_type: VictoryType
    description: str
    requirements: Dict  # Specific requirements for this victory
    value_gained: Dict  # What is gained from this victory
    value_lost: Dict  # What is lost (for pyrrhic victories)
    duration: Optional[int] = None  # How long the victory lasts (days)

class VictoryConditionSystem:
    """Reframes victories as established norms and social positions"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.tracked_victories = {}
    
    def define_victory_condition(self, conflict_id: str, stakeholder_id: str) -> VictoryCondition:
        """Define what victory means for a stakeholder in a conflict"""
        conflict = self._get_conflict(conflict_id)
        stakeholder_goals = self._analyze_stakeholder_goals(stakeholder_id, conflict)
        
        # Generate appropriate victory condition
        if stakeholder_goals['type'] == 'control':
            return self._create_control_victory(stakeholder_id, conflict)
        elif stakeholder_goals['type'] == 'recognition':
            return self._create_recognition_victory(stakeholder_id, conflict)
        elif stakeholder_goals['type'] == 'precedent':
            return self._create_precedent_victory(stakeholder_id, conflict)
        else:
            return self._create_generic_victory(stakeholder_id, conflict)
    
    def _create_control_victory(self, stakeholder_id: str, conflict: Dict) -> VictoryCondition:
        """Create a control-based victory condition"""
        return VictoryCondition(
            condition_id=f"victory_{conflict['id']}_{stakeholder_id}",
            victory_type=VictoryType.NORM_ESTABLISHMENT,
            description=f"Establish control over {conflict['domain']}",
            requirements={
                "acceptance_count": 3,  # Need 3 instances of acceptance
                "resistance_level": "minimal",  # Less than 20% resistance
                "time_elapsed": 7  # Over at least 7 days
            },
            value_gained={
                "authority": f"Recognized authority over {conflict['domain']}",
                "precedent": "Future decisions in this area defer to you",
                "social_capital": 10
            },
            value_lost={},
            duration=30  # Lasts 30 days before can be challenged
        )
    
    def evaluate_victory(self, conflict_id: str, resolution_data: Dict) -> Dict:
        """Evaluate if anyone achieved victory and what kind"""
        victories = []
        
        for stakeholder_id in resolution_data['stakeholders']:
            condition = self.define_victory_condition(conflict_id, stakeholder_id)
            if self._check_victory_achieved(condition, resolution_data):
                victory_result = self._process_victory(stakeholder_id, condition, resolution_data)
                victories.append(victory_result)
        
        # Check for pyrrhic victories
        for victory in victories:
            if self._is_pyrrhic(victory, resolution_data):
                victory['type'] = VictoryType.PYRRHIC
                victory['cost'] = self._calculate_pyrrhic_cost(victory, resolution_data)
        
        return {
            "victories": victories,
            "new_norms": self._extract_new_norms(victories),
            "power_shift": self._calculate_power_shift(victories)
        }
    
    def _check_victory_achieved(self, condition: VictoryCondition, resolution_data: Dict) -> bool:
        """Check if victory requirements are met"""
        for req_key, req_value in condition.requirements.items():
            if req_key not in resolution_data or resolution_data[req_key] < req_value:
                return False
        return True
    
    def _is_pyrrhic(self, victory: Dict, resolution_data: Dict) -> bool:
        """Check if victory came at too great a cost"""
        # Victory is pyrrhic if:
        # - Damaged important relationships
        # - Lost more social capital than gained
        # - Created lasting resentment
        # - Exhausted resources achieving it
        
        relationship_damage = resolution_data.get('relationship_damage', 0)
        resentment_created = resolution_data.get('resentment_level', 0)
        resource_cost = resolution_data.get('resource_expenditure', 0)
        
        pyrrhic_score = (relationship_damage * 2 + resentment_created + resource_cost) / 4
        return pyrrhic_score > 0.6  # Threshold for pyrrhic victory
    
    def establish_comfort_zone(self, npc_id: str, boundaries: List[str]) -> Dict:
        """Establish an NPC's comfort zone as social fact"""
        comfort_zone = {
            "owner_id": npc_id,
            "boundaries": boundaries,
            "established_date": datetime.now(),
            "respect_level": 0.5,  # Starts at neutral
            "violations": []
        }
        
        self.db.execute("""
            INSERT INTO comfort_zones
            (owner_id, boundaries, established_date, respect_level)
            VALUES (?, ?, ?, ?)
        """, (npc_id, json.dumps(boundaries), datetime.now(), 0.5))
        
        return {
            "success": True,
            "comfort_zone": comfort_zone,
            "message": f"Established comfort zone with {len(boundaries)} boundaries",
            "enforcement": "Violations will create automatic friction"
        }
    
    def track_long_term_consequences(self, victory_id: str, days_elapsed: int):
        """Track the long-term consequences of victories"""
        victory = self._get_victory(victory_id)
        
        consequences = {
            "immediate": self._get_immediate_consequences(victory),
            "short_term": self._get_short_term_consequences(victory, days_elapsed),
            "long_term": self._get_long_term_consequences(victory, days_elapsed),
            "relationships": self._get_relationship_consequences(victory, days_elapsed)
        }
        
        # Check if victory benefits are fading
        if days_elapsed > victory.get('duration', 30):
            consequences['status'] = 'fading'
            consequences['challenge_possible'] = True
        
        return consequences

# Database Schemas
POWER_STRUGGLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS domestic_power_struggles (
    struggle_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    instigator_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    current_stage TEXT,
    control_percentage REAL,
    precedents JSON,
    resistance_events JSON,
    started_at TIMESTAMP,
    last_action TIMESTAMP,
    active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS established_dominance (
    controller_id TEXT,
    domain TEXT,
    established_date TIMESTAMP,
    acceptance_level TEXT,
    PRIMARY KEY (controller_id, domain)
);

CREATE TABLE IF NOT EXISTS default_preferences (
    category TEXT PRIMARY KEY,
    preference TEXT NOT NULL,
    controller_id TEXT NOT NULL,
    established_date TIMESTAMP,
    challenge_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS victory_conditions (
    condition_id TEXT PRIMARY KEY,
    victory_type TEXT,
    description TEXT,
    requirements JSON,
    value_gained JSON,
    value_lost JSON,
    duration INTEGER
);

CREATE TABLE IF NOT EXISTS achieved_victories (
    victory_id TEXT PRIMARY KEY,
    stakeholder_id TEXT NOT NULL,
    condition_id TEXT NOT NULL,
    achieved_date TIMESTAMP,
    active_until TIMESTAMP,
    consequences JSON,
    FOREIGN KEY (condition_id) REFERENCES victory_conditions(condition_id)
);

CREATE TABLE IF NOT EXISTS comfort_zones (
    owner_id TEXT PRIMARY KEY,
    boundaries JSON NOT NULL,
    established_date TIMESTAMP,
    respect_level REAL,
    violations JSON
);
"""
