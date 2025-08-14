# Resource Competition & Social Circle Dynamics Systems

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta, time
from enum import Enum
import json
import random
from collections import defaultdict

# RESOURCE COMPETITION SYSTEM

class ResourceType(Enum):
    """Types of resources that can be contested"""
    TIME = "time"                        # Personal time with someone
    ATTENTION = "attention"              # Focus and emotional energy
    SPACE = "space"                      # Physical territory
    COMFORT = "comfort"                  # Comfort items (hot water, best chair)
    DECISION = "decision"                # Decision-making power
    INFORMATION = "information"          # Access to knowledge/secrets
    AFFECTION = "affection"             # Emotional intimacy
    PRIVILEGE = "privilege"             # Special permissions/freedoms

@dataclass
class Resource:
    """A contestable resource"""
    resource_id: str
    resource_type: ResourceType
    name: str
    total_capacity: float = 1.0         # Total available amount
    allocated: Dict[str, float] = field(default_factory=dict)  # Character -> amount
    contested: bool = False
    scarcity_level: float = 0.0        # 0 = abundant, 1 = extremely scarce
    refresh_rate: Optional[str] = None  # 'daily', 'hourly', 'weekly', or None
    last_refresh: datetime = field(default_factory=datetime.now)
    claims: List[Dict] = field(default_factory=list)  # Historical claims
    
    def request_allocation(self, character: str, amount: float) -> Tuple[bool, float]:
        """Request resource allocation"""
        current_allocated = sum(self.allocated.values())
        available = self.total_capacity - current_allocated
        
        if amount <= available:
            # Grant full request
            self.allocated[character] = self.allocated.get(character, 0) + amount
            self.claims.append({
                'character': character,
                'amount': amount,
                'granted': amount,
                'timestamp': datetime.now()
            })
            return True, amount
        elif available > 0:
            # Grant partial
            self.allocated[character] = self.allocated.get(character, 0) + available
            self.claims.append({
                'character': character,
                'amount': amount,
                'granted': available,
                'timestamp': datetime.now()
            })
            return True, available
        else:
            # Denied - resource exhausted
            self.contested = True
            self.claims.append({
                'character': character,
                'amount': amount,
                'granted': 0,
                'timestamp': datetime.now(),
                'reason': 'exhausted'
            })
            return False, 0
    
    def calculate_scarcity(self) -> float:
        """Calculate current scarcity level"""
        demand = len(self.claims)  # Recent claim count
        current_allocated = sum(self.allocated.values())
        utilization = current_allocated / max(0.01, self.total_capacity)
        
        # Combine factors
        self.scarcity_level = min(1.0, (utilization * 0.7 + min(demand * 0.1, 0.3)))
        return self.scarcity_level
    
    def refresh(self):
        """Refresh resource based on refresh rate"""
        if self.refresh_rate == 'hourly':
            self.allocated.clear()
        elif self.refresh_rate == 'daily':
            if (datetime.now() - self.last_refresh).days >= 1:
                self.allocated.clear()
                self.last_refresh = datetime.now()
        elif self.refresh_rate == 'weekly':
            if (datetime.now() - self.last_refresh).days >= 7:
                self.allocated.clear()
                self.last_refresh = datetime.now()
        
        # Clear old claims
        cutoff = datetime.now() - timedelta(hours=24)
        self.claims = [c for c in self.claims if c['timestamp'] > cutoff]

@dataclass 
class AttentionEconomy:
    """Manages attention as a special resource"""
    giver: str  # Who's attention is being competed for
    total_attention: float = 1.0  # Total attention to give
    current_distribution: Dict[str, float] = field(default_factory=dict)
    desired_distribution: Dict[str, float] = field(default_factory=dict)  # What each person wants
    satisfaction_levels: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    
    def distribute_attention(self, time_period: str) -> Dict[str, float]:
        """Distribute attention for a time period"""
        # Calculate based on various factors
        distribution = {}
        
        # Start with equal distribution
        seekers = list(self.desired_distribution.keys())
        if not seekers:
            return {}
            
        base_share = self.total_attention / len(seekers)
        
        for seeker in seekers:
            distribution[seeker] = base_share
            
            # Adjust based on recent satisfaction
            if self.satisfaction_levels.get(seeker, 0.5) < 0.3:
                # Low satisfaction gets boost
                distribution[seeker] *= 1.3
            elif self.satisfaction_levels.get(seeker, 0.5) > 0.8:
                # High satisfaction gets less
                distribution[seeker] *= 0.8
        
        # Normalize to total
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v * self.total_attention / total 
                          for k, v in distribution.items()}
        
        self.current_distribution = distribution
        self._update_satisfaction()
        
        # Record history
        self.history.append({
            'time_period': time_period,
            'distribution': distribution.copy(),
            'timestamp': datetime.now()
        })
        
        return distribution
    
    def _update_satisfaction(self):
        """Update satisfaction based on met vs desired attention"""
        for seeker in self.desired_distribution:
            desired = self.desired_distribution[seeker]
            received = self.current_distribution.get(seeker, 0)
            
            if desired > 0:
                ratio = received / desired
                # Satisfaction changes gradually
                current_sat = self.satisfaction_levels.get(seeker, 0.5)
                new_sat = current_sat * 0.7 + ratio * 0.3  # Weighted average
                self.satisfaction_levels[seeker] = min(1.0, max(0.0, new_sat))
    
    def calculate_tension_from_attention(self) -> Dict[Tuple[str, str], float]:
        """Calculate tension between seekers based on attention competition"""
        tensions = {}
        
        seekers = list(self.desired_distribution.keys())
        for i, seeker_a in enumerate(seekers):
            for seeker_b in seekers[i+1:]:
                # Tension based on satisfaction difference
                sat_a = self.satisfaction_levels.get(seeker_a, 0.5)
                sat_b = self.satisfaction_levels.get(seeker_b, 0.5)
                
                tension = abs(sat_a - sat_b) * 0.5
                
                # Extra tension if both are unsatisfied
                if sat_a < 0.4 and sat_b < 0.4:
                    tension += 0.3
                
                tensions[(seeker_a, seeker_b)] = tension
        
        return tensions

class ResourceCompetitionSystem:
    """Manages all resource competition"""
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.attention_economies: Dict[str, AttentionEconomy] = {}
        self.competition_events: List[Dict] = []
        self.resource_claims: Dict[str, List[str]] = defaultdict(list)  # Character -> resources
        
    def initialize_household_resources(self):
        """Set up standard household resources"""
        
        # Bathroom time (morning/evening)
        self.resources['bathroom_morning'] = Resource(
            resource_id='bathroom_morning',
            resource_type=ResourceType.SPACE,
            name='Morning bathroom slot (6-9 AM)',
            total_capacity=3.0,  # 3 hours
            refresh_rate='daily'
        )
        
        # Hot water
        self.resources['hot_water'] = Resource(
            resource_id='hot_water',
            resource_type=ResourceType.COMFORT,
            name='Hot water for showers',
            total_capacity=2.0,  # Enough for 2 long showers
            refresh_rate='daily'
        )
        
        # TV control
        self.resources['tv_evening'] = Resource(
            resource_id='tv_evening',
            resource_type=ResourceType.DECISION,
            name='Evening TV control',
            total_capacity=1.0,  # Only one can control
            refresh_rate='daily'
        )
        
        # Kitchen time (dinner prep)
        self.resources['kitchen_dinner'] = Resource(
            resource_id='kitchen_dinner',
            resource_type=ResourceType.SPACE,
            name='Kitchen for dinner prep (5-7 PM)',
            total_capacity=2.0,  # 2 people can share
            refresh_rate='daily'
        )
        
        # Quiet workspace
        self.resources['quiet_space'] = Resource(
            resource_id='quiet_space',
            resource_type=ResourceType.SPACE,
            name='Quiet work/reading space',
            total_capacity=1.0,
            refresh_rate='hourly'
        )
    
    def request_resource(self, character: str, resource_id: str, 
                        amount: float = 1.0) -> Dict:
        """Character requests a resource"""
        if resource_id not in self.resources:
            return {'success': False, 'reason': 'Resource not found'}
        
        resource = self.resources[resource_id]
        success, granted = resource.request_allocation(character, amount)
        
        result = {
            'success': success,
            'requested': amount,
            'granted': granted,
            'resource': resource.name,
            'scarcity': resource.calculate_scarcity()
        }
        
        if success:
            self.resource_claims[character].append(resource_id)
            
            if granted < amount:
                result['partial'] = True
                result['message'] = f"Only partial allocation available"
        else:
            # Create competition event
            self.competition_events.append({
                'type': 'resource_denied',
                'character': character,
                'resource': resource_id,
                'competitors': list(resource.allocated.keys()),
                'timestamp': datetime.now()
            })
            result['message'] = "Resource unavailable"
            result['competitors'] = list(resource.allocated.keys())
        
        return result
    
    def negotiate_resource_trade(self, char_a: str, char_b: str,
                                offer: Dict, request: Dict) -> Dict:
        """Negotiate resource trade between characters"""
        negotiation = {
            'proposer': char_a,
            'recipient': char_b,
            'offer': offer,  # What char_a offers
            'request': request,  # What char_a wants
            'timestamp': datetime.now()
        }
        
        # Calculate fairness
        offer_value = self._calculate_resource_value(offer)
        request_value = self._calculate_resource_value(request)
        
        fairness = offer_value / max(request_value, 0.01)
        
        # Acceptance probability based on fairness
        if fairness >= 0.9:
            accepted = random.random() < 0.8
        elif fairness >= 0.7:
            accepted = random.random() < 0.5
        else:
            accepted = random.random() < 0.2
        
        negotiation['fairness'] = fairness
        negotiation['accepted'] = accepted
        
        if accepted:
            # Execute trade
            self._execute_trade(char_a, char_b, offer, request)
            negotiation['result'] = 'accepted'
        else:
            negotiation['result'] = 'rejected'
            
            # Counter-offer possibility
            if fairness > 0.5 and random.random() < 0.4:
                negotiation['counter_offer'] = self._generate_counter_offer(
                    request, offer, fairness
                )
        
        self.competition_events.append(negotiation)
        return negotiation
    
    def _calculate_resource_value(self, resource_package: Dict) -> float:
        """Calculate value of resource package"""
        value = 0.0
        
        for resource_id, amount in resource_package.items():
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                # Value based on scarcity and amount
                base_value = amount
                scarcity_multiplier = 1 + resource.scarcity_level
                value += base_value * scarcity_multiplier
        
        return value
    
    def _execute_trade(self, char_a: str, char_b: str, 
                      offer: Dict, request: Dict):
        """Execute resource trade"""
        # Transfer offered resources from a to b
        for resource_id, amount in offer.items():
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                if char_a in resource.allocated:
                    transfer = min(amount, resource.allocated[char_a])
                    resource.allocated[char_a] -= transfer
                    resource.allocated[char_b] = resource.allocated.get(char_b, 0) + transfer
        
        # Transfer requested resources from b to a
        for resource_id, amount in request.items():
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                if char_b in resource.allocated:
                    transfer = min(amount, resource.allocated[char_b])
                    resource.allocated[char_b] -= transfer
                    resource.allocated[char_a] = resource.allocated.get(char_a, 0) + transfer
    
    def _generate_counter_offer(self, original_request: Dict, 
                               original_offer: Dict, fairness: float) -> Dict:
        """Generate counter-offer"""
        # Adjust quantities to be more fair
        adjustment = 0.8 / fairness  # Bring closer to fair
        
        counter = {
            'request': {k: v * adjustment for k, v in original_request.items()},
            'offer': original_offer  # Keep offer same
        }
        
        return counter

# SOCIAL CIRCLE DYNAMICS

class SocialCircleType(Enum):
    """Types of social circles"""
    HOUSEHOLD = "household"              # People living together
    FRIEND_GROUP = "friend_group"        # Social friends
    WORK_COLLEAGUES = "work_colleagues"  # Professional relationships
    FAMILY = "family"                    # Blood/chosen family
    INTIMATE = "intimate"                # Romantic/sexual partners
    HOBBY_GROUP = "hobby_group"          # Shared interest group
    RIVALS = "rivals"                    # Competitive dynamics

@dataclass
class SocialCircle:
    """A social circle with its own dynamics"""
    circle_id: str
    circle_type: SocialCircleType
    members: Set[str]
    hierarchy: Dict[str, float] = field(default_factory=dict)  # Status levels
    relationships: Dict[Tuple[str, str], Dict] = field(default_factory=dict)
    group_norms: List[str] = field(default_factory=list)
    reputation_scores: Dict[str, float] = field(default_factory=dict)
    recent_events: List[Dict] = field(default_factory=list)
    cohesion: float = 0.5  # Group cohesion level
    
    def add_member(self, character: str, initial_status: float = 0.5):
        """Add member to circle"""
        self.members.add(character)
        self.hierarchy[character] = initial_status
        self.reputation_scores[character] = 0.5
        
        # Establish relationships with existing members
        for member in self.members:
            if member != character:
                key = tuple(sorted([character, member]))
                self.relationships[key] = {
                    'affinity': 0.5,
                    'trust': 0.5,
                    'tension': 0.0
                }
    
    def update_reputation(self, character: str, action: Dict) -> float:
        """Update character's reputation based on action"""
        if character not in self.members:
            return 0.0
        
        current_rep = self.reputation_scores.get(character, 0.5)
        
        # Different actions affect reputation differently
        reputation_effects = {
            'support_member': 0.1,
            'betray_member': -0.3,
            'share_resource': 0.05,
            'hoard_resource': -0.1,
            'defend_group': 0.15,
            'break_norm': -0.2,
            'successful_leadership': 0.2,
            'failed_leadership': -0.15
        }
        
        effect = reputation_effects.get(action['type'], 0)
        
        # Modify based on circle type
        if self.circle_type == SocialCircleType.INTIMATE:
            effect *= 1.5  # Actions matter more in intimate circles
        elif self.circle_type == SocialCircleType.WORK_COLLEAGUES:
            effect *= 0.7  # Professional distance
        
        new_rep = min(1.0, max(0.0, current_rep + effect))
        self.reputation_scores[character] = new_rep
        
        # Record event
        self.recent_events.append({
            'character': character,
            'action': action,
            'reputation_change': effect,
            'timestamp': datetime.now()
        })
        
        # Update cohesion
        self._update_cohesion()
        
        return new_rep
    
    def _update_cohesion(self):
        """Update group cohesion based on internal dynamics"""
        if not self.members:
            return
        
        # Average reputation indicates group harmony
        avg_reputation = sum(self.reputation_scores.values()) / len(self.reputation_scores)
        
        # Tension affects cohesion
        total_tension = sum(rel.get('tension', 0) for rel in self.relationships.values())
        avg_tension = total_tension / max(len(self.relationships), 1)
        
        # Calculate new cohesion
        self.cohesion = (avg_reputation * 0.6 + (1 - avg_tension) * 0.4)
    
    def detect_alliances(self) -> List[Set[str]]:
        """Detect sub-alliances within the circle"""
        alliances = []
        processed = set()
        
        for member in self.members:
            if member in processed:
                continue
                
            alliance = {member}
            
            # Find members with high mutual affinity
            for other in self.members:
                if other != member and other not in processed:
                    key = tuple(sorted([member, other]))
                    if key in self.relationships:
                        if self.relationships[key]['affinity'] > 0.7:
                            alliance.add(other)
            
            if len(alliance) > 1:
                alliances.append(alliance)
                processed.update(alliance)
        
        return alliances
    
    def calculate_influence_network(self) -> Dict[str, float]:
        """Calculate influence each member has in the circle"""
        influence = {}
        
        for member in self.members:
            # Base influence from hierarchy
            base = self.hierarchy.get(member, 0.5)
            
            # Reputation modifier
            rep_modifier = self.reputation_scores.get(member, 0.5)
            
            # Relationship modifier (how many trust them)
            trust_count = 0
            for key, rel in self.relationships.items():
                if member in key:
                    if rel['trust'] > 0.6:
                        trust_count += 1
            
            relationship_modifier = trust_count / max(len(self.members) - 1, 1)
            
            # Calculate total influence
            influence[member] = (base * 0.4 + rep_modifier * 0.3 + 
                                relationship_modifier * 0.3)
        
        return influence

class SocialDynamicsSystem:
    """Manages social circles and their dynamics"""
    
    def __init__(self):
        self.circles: Dict[str, SocialCircle] = {}
        self.character_circles: Dict[str, Set[str]] = defaultdict(set)  # Character -> circles
        self.reputation_ripples: List[Dict] = []  # Track reputation effects
        
    def create_circle(self, circle_type: SocialCircleType, 
                     initial_members: Set[str], norms: List[str] = None) -> SocialCircle:
        """Create a new social circle"""
        circle_id = f"{circle_type.value}_{datetime.now().timestamp()}"
        
        circle = SocialCircle(
            circle_id=circle_id,
            circle_type=circle_type,
            members=set(),
            group_norms=norms or []
        )
        
        # Add initial members
        for member in initial_members:
            circle.add_member(member)
            self.character_circles[member].add(circle_id)
        
        self.circles[circle_id] = circle
        return circle
    
    def process_social_action(self, actor: str, action: Dict, 
                             circle_id: str) -> Dict:
        """Process action within social context"""
        if circle_id not in self.circles:
            return {'error': 'Circle not found'}
        
        circle = self.circles[circle_id]
        
        # Update reputation
        old_rep = circle.reputation_scores.get(actor, 0.5)
        new_rep = circle.update_reputation(actor, action)
        
        # Check for ripple effects
        ripples = self._calculate_reputation_ripples(
            actor, old_rep, new_rep, circle
        )
        
        # Check for norm violations
        norm_violations = self._check_norm_violations(action, circle.group_norms)
        
        result = {
            'actor': actor,
            'action': action,
            'reputation_change': new_rep - old_rep,
            'new_reputation': new_rep,
            'ripple_effects': ripples,
            'norm_violations': norm_violations,
            'circle_cohesion': circle.cohesion
        }
        
        # Check for status changes
        if abs(new_rep - old_rep) > 0.2:
            result['status_change'] = self._calculate_status_change(
                actor, circle
            )
        
        return result
    
    def _calculate_reputation_ripples(self, actor: str, old_rep: float,
                                     new_rep: float, circle: SocialCircle) -> List[Dict]:
        """Calculate how reputation change affects others"""
        ripples = []
        
        change = new_rep - old_rep
        
        # Major reputation changes affect relationships
        if abs(change) > 0.15:
            for key, rel in circle.relationships.items():
                if actor in key:
                    other = [m for m in key if m != actor][0]
                    
                    if change > 0:
                        # Positive change improves relationships
                        rel['affinity'] = min(1.0, rel['affinity'] + change * 0.3)
                        rel['trust'] = min(1.0, rel['trust'] + change * 0.2)
                    else:
                        # Negative change strains relationships
                        rel['trust'] = max(0.0, rel['trust'] + change * 0.4)
                        rel['tension'] = min(1.0, rel['tension'] - change * 0.3)
                    
                    ripples.append({
                        'affected': other,
                        'relationship_change': change * 0.3,
                        'type': 'relationship_impact'
                    })
        
        # Extreme changes affect whole group
        if abs(change) > 0.25:
            ripples.append({
                'affected': 'group',
                'cohesion_change': change * 0.2,
                'type': 'group_impact'
            })
        
        return ripples
    
    def _check_norm_violations(self, action: Dict, norms: List[str]) -> List[str]:
        """Check if action violates group norms"""
        violations = []
        
        norm_keywords = {
            'share_resources': ['hoard', 'refuse', 'monopolize'],
            'support_members': ['betray', 'abandon', 'undermine'],
            'maintain_harmony': ['conflict', 'argue', 'disrupt'],
            'respect_hierarchy': ['challenge_authority', 'insubordination'],
            'keep_secrets': ['reveal', 'gossip', 'expose']
        }
        
        action_str = json.dumps(action).lower()
        
        for norm in norms:
            if norm in norm_keywords:
                for keyword in norm_keywords[norm]:
                    if keyword in action_str:
                        violations.append(norm)
                        break
        
        return violations
    
    def _calculate_status_change(self, character: str, 
                                circle: SocialCircle) -> Dict:
        """Calculate status change from reputation shift"""
        current_status = circle.hierarchy.get(character, 0.5)
        reputation = circle.reputation_scores.get(character, 0.5)
        
        # Status tends toward reputation over time
        new_status = current_status * 0.7 + reputation * 0.3
        
        circle.hierarchy[character] = new_status
        
        # Determine relative position
        ranked = sorted(circle.hierarchy.items(), key=lambda x: x[1], reverse=True)
        position = [i for i, (c, _) in enumerate(ranked) if c == character][0]
        
        return {
            'new_status': new_status,
            'position': position + 1,
            'total_members': len(circle.members),
            'relative': 'rising' if new_status > current_status else 'falling'
        }
    
    def mediate_cross_circle_conflict(self, char_a: str, char_b: str,
                                     conflict: Dict) -> Dict:
        """Handle conflicts that span multiple circles"""
        # Find shared circles
        circles_a = self.character_circles.get(char_a, set())
        circles_b = self.character_circles.get(char_b, set())
        shared_circles = circles_a & circles_b
        
        effects = {}
        
        for circle_id in shared_circles:
            circle = self.circles[circle_id]
            
            # Conflict affects reputation in each circle differently
            if circle.circle_type == SocialCircleType.HOUSEHOLD:
                # Household conflicts are more impactful
                impact_multiplier = 1.5
            elif circle.circle_type == SocialCircleType.WORK_COLLEAGUES:
                # Work tries to stay neutral
                impact_multiplier = 0.5
            else:
                impact_multiplier = 1.0
            
            # Update reputations based on conflict outcome
            if conflict.get('winner'):
                winner = conflict['winner']
                loser = char_b if winner == char_a else char_a
                
                circle.reputation_scores[winner] = min(1.0,
                    circle.reputation_scores.get(winner, 0.5) + 0.1 * impact_multiplier)
                circle.reputation_scores[loser] = max(0.0,
                    circle.reputation_scores.get(loser, 0.5) - 0.1 * impact_multiplier)
            
            # Update relationship
            key = tuple(sorted([char_a, char_b]))
            if key in circle.relationships:
                circle.relationships[key]['tension'] = min(1.0,
                    circle.relationships[key]['tension'] + 0.2)
                circle.relationships[key]['trust'] = max(0.0,
                    circle.relationships[key]['trust'] - 0.15)
            
            effects[circle_id] = {
                'circle_type': circle.circle_type.value,
                'cohesion_impact': -0.1,
                'reputation_changes': {
                    char_a: circle.reputation_scores.get(char_a, 0.5),
                    char_b: circle.reputation_scores.get(char_b, 0.5)
                }
            }
        
        return effects

# Integration Functions

def integrate_resources_with_conflicts(resource_system: ResourceCompetitionSystem,
                                      conflict: Dict) -> Dict:
    """Create conflicts from resource competition"""
    
    if conflict['type'] == 'resource_shortage':
        resource_id = conflict['resource_id']
        resource = resource_system.resources.get(resource_id)
        
        if resource and resource.contested:
            return {
                'conflict_type': 'resource_competition',
                'stakes': resource.name,
                'competitors': list(resource.allocated.keys()),
                'scarcity_level': resource.scarcity_level,
                'suggested_resolutions': [
                    'negotiate_sharing',
                    'trade_for_other_resource',
                    'establish_schedule',
                    'winner_takes_all'
                ]
            }
    
    return {}

def calculate_attention_tensions(attention_economy: AttentionEconomy,
                                tension_system) -> None:
    """Add attention-based tensions to tension system"""
    
    tensions = attention_economy.calculate_tension_from_attention()
    
    for (char_a, char_b), tension_level in tensions.items():
        if tension_level > 0.3:  # Significant tension
            friction = FrictionPoint(
                source=TensionSource.ATTENTION_IMBALANCE,
                intensity=tension_level,
                participants={char_a, char_b},
                context={
                    'competition_for': attention_economy.giver,
                    'satisfaction_a': attention_economy.satisfaction_levels.get(char_a, 0.5),
                    'satisfaction_b': attention_economy.satisfaction_levels.get(char_b, 0.5)
                }
            )
            
            # Add to tension system
            key = tuple(sorted([char_a, char_b]))
            if key not in tension_system.accumulators:
                tension_system.accumulators[key] = TensionAccumulator(char_a, char_b)
            tension_system.accumulators[key].add_friction(friction)

# Database Schema

RESOURCE_SOCIAL_SCHEMA = """
-- Resources Table
CREATE TABLE resources (
    resource_id TEXT PRIMARY KEY,
    resource_type TEXT NOT NULL,
    name TEXT NOT NULL,
    total_capacity REAL DEFAULT 1.0,
    refresh_rate TEXT,
    last_refresh TIMESTAMP,
    contested BOOLEAN DEFAULT FALSE,
    scarcity_level REAL DEFAULT 0.0
);

-- Resource Allocations Table
CREATE TABLE resource_allocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id TEXT REFERENCES resources(resource_id),
    character TEXT NOT NULL,
    amount REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resource Claims History
CREATE TABLE resource_claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id TEXT REFERENCES resources(resource_id),
    character TEXT NOT NULL,
    requested REAL NOT NULL,
    granted REAL NOT NULL,
    success BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Attention Economy Table
CREATE TABLE attention_economy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    giver TEXT NOT NULL,
    receiver TEXT NOT NULL,
    attention_amount REAL,
    satisfaction REAL,
    time_period TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Social Circles Table
CREATE TABLE social_circles (
    circle_id TEXT PRIMARY KEY,
    circle_type TEXT NOT NULL,
    cohesion REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Circle Memberships Table
CREATE TABLE circle_memberships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circle_id TEXT REFERENCES social_circles(circle_id),
    character TEXT NOT NULL,
    status_level REAL DEFAULT 0.5,
    reputation REAL DEFAULT 0.5,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(circle_id, character)
);

-- Circle Relationships Table
CREATE TABLE circle_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circle_id TEXT REFERENCES social_circles(circle_id),
    character_a TEXT NOT NULL,
    character_b TEXT NOT NULL,
    affinity REAL DEFAULT 0.5,
    trust REAL DEFAULT 0.5,
    tension REAL DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(circle_id, character_a, character_b)
);

-- Social Events Table
CREATE TABLE social_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circle_id TEXT REFERENCES social_circles(circle_id),
    actor TEXT NOT NULL,
    action_type TEXT NOT NULL,
    reputation_change REAL,
    details JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# LLM Tool Functions

def check_resource_availability(character: str, resource_id: str,
                               resource_system: ResourceCompetitionSystem) -> str:
    """LLM tool to check resource availability"""
    
    if resource_id not in resource_system.resources:
        return json.dumps({'error': 'Resource not found'})
    
    resource = resource_system.resources[resource_id]
    allocated = sum(resource.allocated.values())
    available = resource.total_capacity - allocated
    
    return json.dumps({
        'resource': resource.name,
        'available': available,
        'total_capacity': resource.total_capacity,
        'current_holders': list(resource.allocated.keys()),
        'scarcity': resource.scarcity_level,
        'contested': resource.contested,
        'can_request': available > 0
    })

def get_social_standing(character: str, 
                       social_system: SocialDynamicsSystem) -> str:
    """LLM tool to get character's social standing"""
    
    standing = {
        'circles': [],
        'total_reputation': 0.0,
        'influence_network': {},
        'alliances': []
    }
    
    for circle_id in social_system.character_circles.get(character, []):
        circle = social_system.circles[circle_id]
        
        circle_info = {
            'circle_type': circle.circle_type.value,
            'reputation': circle.reputation_scores.get(character, 0.5),
            'status': circle.hierarchy.get(character, 0.5),
            'cohesion': circle.cohesion
        }
        
        # Get position in hierarchy
        ranked = sorted(circle.hierarchy.items(), key=lambda x: x[1], reverse=True)
        position = [i for i, (c, _) in enumerate(ranked) if c == character][0]
        circle_info['position'] = f"{position + 1} of {len(circle.members)}"
        
        standing['circles'].append(circle_info)
        standing['total_reputation'] += circle_info['reputation']
        
        # Check for alliances
        alliances = circle.detect_alliances()
        for alliance in alliances:
            if character in alliance:
                standing['alliances'].append({
                    'circle': circle.circle_type.value,
                    'allies': list(alliance - {character})
                })
    
    if standing['circles']:
        standing['average_reputation'] = standing['total_reputation'] / len(standing['circles'])
    
    return json.dumps(standing, indent=2)
