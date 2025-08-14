# Dynamic Conflict Template System
# Generates context-aware conflicts based on personality, mood, and time

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
from datetime import datetime

class ConflictCategory(Enum):
    DOMESTIC_CONTROL = "domestic_control"
    ATTENTION_SEEKING = "attention_seeking"
    BOUNDARY_TESTING = "boundary_testing"
    RESOURCE_CLAIM = "resource_claim"
    PRECEDENT_SETTING = "precedent_setting"
    SOCIAL_POSITIONING = "social_positioning"

@dataclass
class ConflictTemplate:
    """Template for generating dynamic conflicts"""
    template_id: str
    category: ConflictCategory
    personality_triggers: Dict[str, float]  # personality_trait: likelihood_modifier
    mood_modifiers: Dict[str, float]  # mood: intensity_modifier
    time_windows: List[Tuple[int, int]]  # [(start_hour, end_hour), ...]
    base_intensity: float
    escalation_pattern: str  # "gradual", "sudden", "cyclic"
    required_context: List[str]  # ["alone_together", "others_present", etc]
    
class DynamicTemplateSystem:
    def __init__(self, db_connection):
        self.db = db_connection
        self.templates = self._load_templates()
        
    def _load_templates(self) -> List[ConflictTemplate]:
        """Load conflict templates from database or define them"""
        templates = [
            ConflictTemplate(
                template_id="morning_routine_dominance",
                category=ConflictCategory.DOMESTIC_CONTROL,
                personality_triggers={
                    "controlling": 1.5,
                    "passive_aggressive": 1.3,
                    "competitive": 1.2
                },
                mood_modifiers={
                    "irritated": 1.4,
                    "tired": 1.2,
                    "energetic": 0.8
                },
                time_windows=[(6, 10)],  # Morning hours
                base_intensity=0.3,
                escalation_pattern="gradual",
                required_context=["morning_routine", "shared_space"]
            ),
            ConflictTemplate(
                template_id="evening_attention_competition",
                category=ConflictCategory.ATTENTION_SEEKING,
                personality_triggers={
                    "needy": 1.6,
                    "jealous": 1.4,
                    "insecure": 1.3
                },
                mood_modifiers={
                    "lonely": 1.5,
                    "bored": 1.3,
                    "content": 0.6
                },
                time_windows=[(18, 22)],  # Evening hours
                base_intensity=0.4,
                escalation_pattern="cyclic",
                required_context=["multiple_npcs_present"]
            ),
            ConflictTemplate(
                template_id="weekend_territory_claim",
                category=ConflictCategory.RESOURCE_CLAIM,
                personality_triggers={
                    "territorial": 1.7,
                    "possessive": 1.5,
                    "dominant": 1.3
                },
                mood_modifiers={
                    "relaxed": 0.7,
                    "stressed": 1.4
                },
                time_windows=[(9, 18)],  # Daytime weekend
                base_intensity=0.5,
                escalation_pattern="sudden",
                required_context=["weekend", "shared_activity"]
            )
        ]
        return templates
    
    def generate_conflict(self, context: Dict) -> Optional[Dict]:
        """Generate a conflict based on current context"""
        current_hour = context.get('hour', 12)
        day_type = context.get('day_type', 'weekday')
        npcs_present = context.get('npcs_present', [])
        location = context.get('location', 'home')
        
        # Filter templates by time and context
        valid_templates = []
        for template in self.templates:
            if self._is_template_valid(template, current_hour, context):
                score = self._calculate_template_score(template, npcs_present, context)
                if score > 0.5:  # Threshold for conflict generation
                    valid_templates.append((template, score))
        
        if not valid_templates:
            return None
            
        # Select template weighted by score
        template, score = self._weighted_selection(valid_templates)
        
        # Generate specific conflict from template
        return self._instantiate_conflict(template, context, score)
    
    def _is_template_valid(self, template: ConflictTemplate, hour: int, context: Dict) -> bool:
        """Check if template is valid for current context"""
        # Check time window
        time_valid = any(start <= hour < end for start, end in template.time_windows)
        if not time_valid:
            return False
            
        # Check required context
        for req in template.required_context:
            if req == "weekend" and context.get('day_type') != 'weekend':
                return False
            elif req == "multiple_npcs_present" and len(context.get('npcs_present', [])) < 2:
                return False
            elif req == "alone_together" and len(context.get('npcs_present', [])) != 1:
                return False
                
        return True
    
    def _calculate_template_score(self, template: ConflictTemplate, npcs: List, context: Dict) -> float:
        """Calculate likelihood score for this template"""
        score = template.base_intensity
        
        # Apply personality modifiers
        for npc in npcs:
            personality = self._get_npc_personality(npc['id'])
            for trait, modifier in template.personality_triggers.items():
                if trait in personality:
                    score *= modifier * personality[trait]
        
        # Apply mood modifiers
        for npc in npcs:
            mood = self._get_npc_mood(npc['id'])
            if mood in template.mood_modifiers:
                score *= template.mood_modifiers[mood]
        
        # Apply tension accumulation
        tension_level = self._get_current_tension(context.get('player_id'))
        score *= (1 + tension_level * 0.5)
        
        # Apply recent conflict cooldown
        last_conflict_hours = self._hours_since_last_conflict(context.get('player_id'))
        if last_conflict_hours < 4:
            score *= 0.3  # Reduce likelihood if recent conflict
        elif last_conflict_hours > 24:
            score *= 1.3  # Increase if no recent conflicts
            
        return min(score, 1.0)  # Cap at 1.0
    
    def _instantiate_conflict(self, template: ConflictTemplate, context: Dict, score: float) -> Dict:
        """Create specific conflict instance from template"""
        npcs = context.get('npcs_present', [])
        primary_npc = self._select_primary_stakeholder(npcs, template)
        
        # Generate conflict details based on template category
        if template.category == ConflictCategory.DOMESTIC_CONTROL:
            conflict = self._generate_domestic_control_conflict(primary_npc, template, context)
        elif template.category == ConflictCategory.ATTENTION_SEEKING:
            conflict = self._generate_attention_conflict(primary_npc, template, context)
        elif template.category == ConflictCategory.RESOURCE_CLAIM:
            conflict = self._generate_resource_conflict(primary_npc, template, context)
        else:
            conflict = self._generate_generic_conflict(primary_npc, template, context)
        
        # Set intensity based on score and escalation pattern
        conflict['intensity'] = self._calculate_intensity(score, template.escalation_pattern)
        conflict['template_id'] = template.template_id
        conflict['auto_generated'] = True
        
        return conflict
    
    def _generate_domestic_control_conflict(self, npc: Dict, template: ConflictTemplate, context: Dict) -> Dict:
        """Generate a domestic control conflict"""
        control_aspects = [
            {"issue": "morning_schedule", "stake": "who decides when breakfast happens"},
            {"issue": "space_usage", "stake": "who gets to use the bathroom first"},
            {"issue": "noise_levels", "stake": "who determines acceptable volume"},
            {"issue": "temperature", "stake": "who controls the thermostat"},
            {"issue": "cleanliness", "stake": "whose standards are followed"}
        ]
        
        aspect = random.choice(control_aspects)
        
        return {
            "type": "domestic_dominance",
            "primary_stakeholder_id": npc['id'],
            "issue": aspect['issue'],
            "description": f"{npc['name']} is asserting control over {aspect['stake']}",
            "stakes": {
                "immediate": f"Control over {aspect['issue']}",
                "precedent": f"Establishing who decides {aspect['stake']}",
                "relationship": f"Power balance in domestic decisions"
            },
            "available_responses": self._generate_response_options(template.category)
        }
    
    def _generate_response_options(self, category: ConflictCategory) -> List[Dict]:
        """Generate appropriate response options based on conflict category"""
        if category == ConflictCategory.DOMESTIC_CONTROL:
            return [
                {"action": "comply", "description": "Accept their control", "outcome": "reinforces_hierarchy"},
                {"action": "negotiate", "description": "Propose a compromise", "outcome": "shared_control"},
                {"action": "resist", "description": "Assert your preference", "outcome": "power_struggle"},
                {"action": "deflect", "description": "Make it seem unimportant", "outcome": "tension_remains"},
                {"action": "subvert", "description": "Agree but do it your way later", "outcome": "hidden_rebellion"}
            ]
        # Add more categories as needed
        return []
    
    def adapt_difficulty(self, player_id: str, conflict_id: str, player_response: str):
        """Adapt future conflict difficulty based on player engagement"""
        # Track player response patterns
        self.db.execute("""
            INSERT INTO conflict_responses 
            (player_id, conflict_id, response_type, timestamp)
            VALUES (?, ?, ?, ?)
        """, (player_id, conflict_id, player_response, datetime.now()))
        
        # Analyze response patterns
        recent_responses = self._get_recent_responses(player_id, days=7)
        
        # Adjust future conflict generation
        if self._player_avoiding_conflicts(recent_responses):
            self._reduce_conflict_frequency(player_id)
        elif self._player_always_escalating(recent_responses):
            self._increase_conflict_complexity(player_id)
        elif self._player_seeking_variety(recent_responses):
            self._diversify_conflict_types(player_id)
    
    def _get_npc_personality(self, npc_id: str) -> Dict[str, float]:
        """Get NPC personality traits"""
        # Query from database or personality system
        return {}
    
    def _get_npc_mood(self, npc_id: str) -> str:
        """Get current NPC mood"""
        # Query from mood system
        return "neutral"
    
    def _get_current_tension(self, player_id: str) -> float:
        """Get accumulated tension level"""
        # Query from tension system
        return 0.5
    
    def _hours_since_last_conflict(self, player_id: str) -> int:
        """Calculate hours since last conflict"""
        # Query from conflict history
        return 12
    
    def _weighted_selection(self, templates: List[Tuple[ConflictTemplate, float]]) -> Tuple[ConflictTemplate, float]:
        """Select template weighted by scores"""
        if not templates:
            return None, 0
        
        weights = [score for _, score in templates]
        selected = random.choices(templates, weights=weights, k=1)[0]
        return selected

# Database schema additions
CONFLICT_TEMPLATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS conflict_templates (
    template_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    personality_triggers JSON,
    mood_modifiers JSON,
    time_windows JSON,
    base_intensity REAL,
    escalation_pattern TEXT,
    required_context JSON,
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.5,
    last_used TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conflict_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    conflict_id TEXT NOT NULL,
    response_type TEXT,
    timestamp TIMESTAMP,
    outcome_satisfaction REAL,
    FOREIGN KEY (conflict_id) REFERENCES conflicts(id)
);

CREATE TABLE IF NOT EXISTS conflict_adaptation (
    player_id TEXT PRIMARY KEY,
    frequency_modifier REAL DEFAULT 1.0,
    complexity_preference TEXT DEFAULT 'medium',
    preferred_categories JSON,
    avoided_categories JSON,
    last_updated TIMESTAMP
);
"""
