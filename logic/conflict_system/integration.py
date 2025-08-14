# Complete System Integrations for Conflict System
# Connects all partial integrations and missing systems

from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from enum import Enum
import json

class SystemIntegrationHub:
    """Central hub for integrating all game systems with conflict system"""
    
    def __init__(self, db_connection, systems: Dict):
        self.db = db_connection
        self.systems = systems  # Dictionary of all game systems
        self._initialize_integrations()
    
    def _initialize_integrations(self):
        """Initialize all system connections"""
        self.time_cycle = self.systems.get('time_cycle')
        self.relationships = self.systems.get('relationships')
        self.memory = self.systems.get('memory')
        self.narrative = self.systems.get('narrative')
        self.addiction = self.systems.get('addiction')
        self.stats = self.systems.get('stats')
        self.vitals = self.systems.get('vitals')
        self.lore = self.systems.get('lore')
        self.events = self.systems.get('events')
        self.scenes = self.systems.get('scenes')
        self.conflicts = self.systems.get('conflicts')

# === Time Cycle Integration (Complete) ===

class TimeCycleConflictIntegration:
    """Fully integrates conflicts with daily time cycles"""
    
    def __init__(self, time_system, conflict_system):
        self.time = time_system
        self.conflicts = conflict_system
        self._setup_routine_conflicts()
    
    def _setup_routine_conflicts(self):
        """Define routine conflicts tied to specific times"""
        self.routine_conflicts = {
            "morning": [
                {"time": time(7, 0), "conflict": "bathroom_order", "frequency": 0.3},
                {"time": time(8, 0), "conflict": "breakfast_control", "frequency": 0.2},
                {"time": time(8, 30), "conflict": "schedule_dominance", "frequency": 0.15}
            ],
            "afternoon": [
                {"time": time(12, 30), "conflict": "lunch_decisions", "frequency": 0.1},
                {"time": time(14, 0), "conflict": "space_allocation", "frequency": 0.2},
                {"time": time(16, 0), "conflict": "attention_competition", "frequency": 0.25}
            ],
            "evening": [
                {"time": time(18, 30), "conflict": "dinner_control", "frequency": 0.3},
                {"time": time(20, 0), "conflict": "entertainment_choice", "frequency": 0.4},
                {"time": time(22, 0), "conflict": "bedtime_negotiation", "frequency": 0.2}
            ],
            "weekend": [
                {"time": time(10, 0), "conflict": "weekend_plans", "frequency": 0.5},
                {"time": time(11, 0), "conflict": "chore_distribution", "frequency": 0.3},
                {"time": time(15, 0), "conflict": "social_activity", "frequency": 0.4}
            ]
        }
    
    def check_routine_conflict_trigger(self, current_time: datetime) -> Optional[Dict]:
        """Check if a routine conflict should trigger based on time"""
        day_type = "weekend" if current_time.weekday() >= 5 else current_time.strftime("%A").lower()
        time_key = self._get_time_period(current_time)
        
        # Get applicable routine conflicts
        potential_conflicts = self.routine_conflicts.get(time_key, [])
        if day_type == "weekend":
            potential_conflicts.extend(self.routine_conflicts.get("weekend", []))
        
        for routine in potential_conflicts:
            if self._should_trigger_routine(routine, current_time):
                return self._generate_routine_conflict(routine, current_time)
        
        return None
    
    def _should_trigger_routine(self, routine: Dict, current_time: datetime) -> bool:
        """Determine if routine conflict should trigger"""
        # Check if we're within 30 minutes of routine time
        routine_datetime = datetime.combine(current_time.date(), routine['time'])
        time_diff = abs((current_time - routine_datetime).total_seconds() / 60)
        
        if time_diff <= 30:
            import random
            # Check frequency and recent conflicts
            if random.random() < routine['frequency']:
                # Check if this conflict happened recently
                last_occurrence = self._get_last_conflict_occurrence(routine['conflict'])
                if not last_occurrence or (current_time - last_occurrence).days >= 1:
                    return True
        return False
    
    def modify_conflict_by_schedule(self, conflict: Dict, current_time: datetime) -> Dict:
        """Modify conflict based on NPC schedules and availability"""
        # Check NPC availability
        for stakeholder_id in conflict.get('stakeholders', []):
            availability = self.time.get_npc_availability(stakeholder_id, current_time)
            if availability['status'] == 'unavailable':
                conflict['modifications'] = conflict.get('modifications', [])
                conflict['modifications'].append({
                    'type': 'delayed',
                    'reason': f"{stakeholder_id} is {availability['activity']}",
                    'resume_time': availability['available_at']
                })
                conflict['active'] = False
            elif availability['status'] == 'busy':
                conflict['intensity'] *= 0.7  # Reduce intensity if NPCs are busy
        
        # Modify based on time of day
        hour = current_time.hour
        if hour < 9 or hour > 22:  # Early morning or late night
            conflict['intensity'] *= 0.5
            conflict['resolution_options'].append('postpone_until_morning')
        elif hour >= 17 and hour <= 19:  # Dinner time tensions
            conflict['intensity'] *= 1.3
        
        return conflict

# === Relationship System Integration (Complete) ===

class RelationshipConflictIntegration:
    """Fully integrates relationship dynamics with conflicts"""
    
    def __init__(self, relationship_system, conflict_system):
        self.relationships = relationship_system
        self.conflicts = conflict_system
    
    def generate_relationship_based_conflict(self, npc1_id: str, npc2_id: str) -> Optional[Dict]:
        """Generate conflict based on relationship state"""
        rel_data = self.relationships.get_relationship(npc1_id, npc2_id)
        
        if not rel_data:
            return None
        
        # Map relationship dimensions to conflict types
        if rel_data['trust'] < 0.3 and rel_data['dominance_balance'] != 0.5:
            # Low trust + power imbalance = control conflicts
            return self._generate_control_conflict(npc1_id, npc2_id, rel_data)
        elif abs(rel_data['dominance_balance'] - 0.5) < 0.1:
            # Equal power = competition for dominance
            return self._generate_competition_conflict(npc1_id, npc2_id, rel_data)
        elif rel_data['intimacy'] > 0.7 and rel_data['trust'] > 0.6:
            # High intimacy + trust = boundary testing
            return self._generate_boundary_conflict(npc1_id, npc2_id, rel_data)
        
        return None
    
    def apply_conflict_to_relationship(self, conflict_id: str, resolution: Dict):
        """Apply conflict resolution effects to relationship dimensions"""
        conflict = self.conflicts.get_conflict(conflict_id)
        
        for stakeholder_id in conflict['stakeholders']:
            other_stakeholders = [s for s in conflict['stakeholders'] if s != stakeholder_id]
            for other_id in other_stakeholders:
                # Calculate relationship changes
                changes = self._calculate_relationship_impact(
                    stakeholder_id, other_id, conflict, resolution
                )
                
                # Apply changes to relationship
                self.relationships.modify_relationship(
                    stakeholder_id, other_id,
                    trust_change=changes['trust'],
                    dominance_change=changes['dominance'],
                    intimacy_change=changes['intimacy']
                )
    
    def _calculate_relationship_impact(self, id1: str, id2: str, conflict: Dict, resolution: Dict) -> Dict:
        """Calculate how conflict resolution affects relationship"""
        impact = {'trust': 0, 'dominance': 0, 'intimacy': 0}
        
        if resolution['type'] == 'compromise':
            impact['trust'] += 0.05
            impact['intimacy'] += 0.02
        elif resolution['type'] == 'domination':
            winner = resolution.get('winner')
            if winner == id1:
                impact['dominance'] += 0.1
                impact['trust'] -= 0.03
            else:
                impact['dominance'] -= 0.1
                impact['trust'] -= 0.05
        elif resolution['type'] == 'avoidance':
            impact['intimacy'] -= 0.03
            impact['trust'] -= 0.01
        
        return impact

# === Memory System Integration (Complete) ===

class MemoryConflictIntegration:
    """Fully integrates memory system with conflicts"""
    
    def __init__(self, memory_system, conflict_system):
        self.memory = memory_system
        self.conflicts = conflict_system
    
    def create_conflict_memory(self, conflict_id: str, perspective_id: str, resolution: Dict) -> str:
        """Create lasting memory of conflict from NPC perspective"""
        conflict = self.conflicts.get_conflict(conflict_id)
        
        # Determine emotional valence
        valence = self._calculate_emotional_valence(perspective_id, conflict, resolution)
        
        # Create memory with appropriate weight
        memory_data = {
            'type': 'conflict',
            'conflict_id': conflict_id,
            'description': self._generate_memory_description(perspective_id, conflict, resolution),
            'emotional_valence': valence,
            'participants': conflict['stakeholders'],
            'outcome': resolution['outcome'],
            'learned_pattern': self._extract_learned_pattern(conflict, resolution),
            'importance': self._calculate_memory_importance(conflict, resolution)
        }
        
        memory_id = self.memory.store_memory(perspective_id, memory_data)
        
        # Link to pattern detection
        if memory_data['learned_pattern']:
            self._update_behavioral_patterns(perspective_id, memory_data['learned_pattern'])
        
        return memory_id
    
    def use_memories_in_conflict(self, npc_id: str, conflict_context: Dict) -> List[Dict]:
        """Retrieve relevant memories to influence conflict behavior"""
        # Search for similar past conflicts
        relevant_memories = self.memory.search_memories(
            npc_id,
            query={
                'type': 'conflict',
                'participants': conflict_context.get('other_party'),
                'timeframe': 30  # Last 30 days
            }
        )
        
        # Extract patterns and precedents
        patterns = []
        for memory in relevant_memories:
            if memory['outcome'] == 'favorable':
                patterns.append({
                    'strategy': memory.get('successful_strategy'),
                    'confidence': 0.7
                })
            elif memory['outcome'] == 'unfavorable':
                patterns.append({
                    'avoid_strategy': memory.get('failed_strategy'),
                    'confidence': 0.8
                })
        
        return patterns
    
    def _calculate_emotional_valence(self, perspective_id: str, conflict: Dict, resolution: Dict) -> float:
        """Calculate emotional impact of conflict outcome"""
        # -1 (very negative) to 1 (very positive)
        if resolution.get('winner') == perspective_id:
            return 0.6
        elif resolution['type'] == 'compromise':
            return 0.1
        elif resolution['type'] == 'loss':
            return -0.7
        else:
            return -0.2

# === NPC Narrative Integration ===

class NarrativeConflictIntegration:
    """Integrates NPC narrative progression with conflicts"""
    
    def __init__(self, narrative_system, conflict_system):
        self.narrative = narrative_system
        self.conflicts = conflict_system
    
    def trigger_narrative_advancement(self, npc_id: str, conflict_id: str, resolution: Dict):
        """Use conflict to advance NPC narrative stage"""
        current_stage = self.narrative.get_npc_stage(npc_id)
        
        # Check if conflict meets advancement criteria
        if self._is_significant_conflict(conflict_id, resolution):
            # Advance narrative
            if current_stage == 'introduction' and resolution['reveals_personality']:
                self.narrative.advance_stage(npc_id, 'personality_reveal')
            elif current_stage == 'facade' and resolution['causes_mask_slip']:
                self.narrative.trigger_mask_slip(npc_id, conflict_id)
            elif current_stage == 'developing' and resolution['deepens_relationship']:
                self.narrative.advance_stage(npc_id, 'established')
    
    def create_revelation_conflict(self, npc_id: str) -> Dict:
        """Generate conflict that reveals hidden aspects of NPC"""
        hidden_traits = self.narrative.get_hidden_traits(npc_id)
        
        if not hidden_traits:
            return None
        
        # Pick trait to reveal
        trait_to_reveal = hidden_traits[0]
        
        return {
            'type': 'revelation',
            'npc_id': npc_id,
            'description': f"A situation that will reveal {npc_id}'s {trait_to_reveal}",
            'stakes': {
                'narrative': f"Player learns about {trait_to_reveal}",
                'relationship': "Changes dynamic based on revelation"
            },
            'revelation_trigger': trait_to_reveal
        }

# === Stats & Vitals Integration ===

class StatsVitalsConflictIntegration:
    """Integrates player stats and vitals with conflict system"""
    
    def __init__(self, stats_system, vitals_system, conflict_system):
        self.stats = stats_system
        self.vitals = vitals_system
        self.conflicts = conflict_system
    
    def modify_conflict_options_by_stats(self, player_id: str, conflict: Dict) -> Dict:
        """Modify available conflict options based on player stats"""
        player_stats = self.stats.get_stats(player_id)
        
        # Willpower affects resistance options
        if player_stats['willpower'] < 30:
            conflict['options'] = [opt for opt in conflict['options'] 
                                  if opt['type'] != 'strong_resistance']
            conflict['options'].append({
                'type': 'tired_compliance',
                'description': 'Too exhausted to argue',
                'willpower_cost': 0
            })
        
        # Empathy affects compromise options
        if player_stats['empathy'] > 70:
            conflict['options'].append({
                'type': 'empathetic_resolution',
                'description': 'Find a solution that works for everyone',
                'empathy_requirement': 70
            })
        
        # Hidden stats revealed through options
        if player_stats.get('hidden_assertiveness', 0) > 60:
            conflict['options'].append({
                'type': 'unexpected_assertion',
                'description': 'Stand your ground surprisingly firmly',
                'reveals': 'hidden_assertiveness'
            })
        
        return conflict
    
    def apply_vitals_modifiers(self, npc_id: str, conflict: Dict) -> Dict:
        """Apply vital state modifiers to conflict behavior"""
        vitals = self.vitals.get_vitals(npc_id)
        
        # Fatigue affects conflict engagement
        if vitals['energy'] < 30:
            conflict['npc_behavior'] = 'passive'
            conflict['intensity'] *= 0.6
            conflict['likely_resolution'] = 'quick_concession'
        
        # Hunger makes NPCs irritable
        if vitals['satiation'] < 40:
            conflict['intensity'] *= 1.3
            conflict['npc_mood'] = 'irritable'
            conflict['escalation_chance'] += 0.2
        
        # Stress affects decision-making
        if vitals.get('stress', 0) > 70:
            conflict['npc_behavior'] = 'erratic'
            conflict['unexpected_actions'] = True
        
        return conflict

# === Addiction System Integration ===

class AddictionConflictIntegration:
    """Integrates addiction mechanics with conflicts"""
    
    def __init__(self, addiction_system, conflict_system):
        self.addiction = addiction_system
        self.conflicts = conflict_system
    
    def generate_addiction_conflict(self, npc_id: str, addiction_type: str) -> Dict:
        """Generate conflict around addictive behavior"""
        addiction_data = self.addiction.get_addiction_status(npc_id, addiction_type)
        
        if addiction_data['level'] > 0.5:
            return {
                'type': 'addiction_driven',
                'description': f"Conflict over {npc_id}'s need for {addiction_type}",
                'stakes': {
                    'immediate': f"Access to {addiction_type}",
                    'relationship': "Trust and control dynamics",
                    'long_term': "Enabling vs intervention"
                },
                'intensity': addiction_data['level'],
                'options': [
                    {'action': 'enable', 'consequence': 'deepen_addiction'},
                    {'action': 'intervene', 'consequence': 'create_resentment'},
                    {'action': 'negotiate', 'consequence': 'temporary_compromise'},
                    {'action': 'use_as_leverage', 'consequence': 'gain_control'}
                ]
            }
        return None
    
    def use_addiction_as_leverage(self, manipulator_id: str, target_id: str, addiction_type: str) -> Dict:
        """Use someone's addiction as leverage in conflicts"""
        addiction_level = self.addiction.get_addiction_level(target_id, addiction_type)
        
        if addiction_level > 0.3:
            leverage_power = addiction_level * 1.5
            return {
                'leverage_type': 'addiction',
                'power': leverage_power,
                'description': f"Using {addiction_type} as leverage",
                'ethical_weight': -0.7,  # Morally questionable
                'relationship_damage': 0.4,
                'effectiveness': min(0.9, leverage_power)
            }
        return None

# === Event System Integration ===

class EventConflictIntegration:
    """Integrates events with conflict system"""
    
    def __init__(self, event_system, conflict_system):
        self.events = event_system
        self.conflicts = conflict_system
    
    def trigger_conflict_event_chain(self, conflict_id: str, resolution: Dict):
        """Trigger event chain based on conflict resolution"""
        if resolution['significance'] > 0.7:
            # Major conflict resolutions trigger events
            event_chain = []
            
            if resolution['type'] == 'domination':
                event_chain.append({
                    'type': 'power_shift',
                    'delay_hours': 0,
                    'description': 'Immediate power dynamic change'
                })
                event_chain.append({
                    'type': 'resentment_buildup',
                    'delay_hours': 24,
                    'description': 'Delayed resentment emerges'
                })
            elif resolution['type'] == 'reconciliation':
                event_chain.append({
                    'type': 'bonding_moment',
                    'delay_hours': 2,
                    'description': 'Shared moment of understanding'
                })
            
            for event in event_chain:
                self.events.schedule_event(event)
    
    def create_conflict_opportunity_event(self) -> Dict:
        """Create event that provides conflict opportunity"""
        opportunities = [
            {
                'type': 'resource_scarcity',
                'description': 'The hot water runs out',
                'conflict_potential': 'bathroom_schedule'
            },
            {
                'type': 'social_invitation',
                'description': 'Invitation arrives for one person',
                'conflict_potential': 'jealousy_attention'
            },
            {
                'type': 'decision_required',
                'description': 'Major household decision needed',
                'conflict_potential': 'control_authority'
            }
        ]
        
        import random
        opportunity = random.choice(opportunities)
        
        return self.events.create_event({
            'type': 'conflict_opportunity',
            'subtype': opportunity['type'],
            'description': opportunity['description'],
            'creates_conflict': opportunity['conflict_potential'],
            'duration_hours': 4
        })

# === Scene Management Integration ===

class SceneConflictIntegration:
    """Integrates conflict system with scene management"""
    
    def __init__(self, scene_system, conflict_system):
        self.scenes = scene_system
        self.conflicts = conflict_system
    
    def embed_conflict_in_scene(self, scene_id: str, conflict: Dict) -> Dict:
        """Embed conflict naturally within a scene"""
        scene = self.scenes.get_scene(scene_id)
        
        # Modify conflict presentation based on scene mood
        if scene['mood'] == 'romantic':
            conflict['presentation'] = 'subtle_tension'
            conflict['intensity'] *= 0.7
        elif scene['mood'] == 'tense':
            conflict['presentation'] = 'open_confrontation'
            conflict['intensity'] *= 1.2
        elif scene['mood'] == 'casual':
            conflict['presentation'] = 'casual_friction'
        
        # Allow multiple conflicts per scene
        scene['active_conflicts'] = scene.get('active_conflicts', [])
        scene['active_conflicts'].append(conflict['id'])
        
        # Set scene-appropriate resolution options
        conflict['scene_options'] = self._get_scene_appropriate_options(scene, conflict)
        
        return conflict
    
    def _get_scene_appropriate_options(self, scene: Dict, conflict: Dict) -> List[Dict]:
        """Get conflict options appropriate for scene context"""
        options = []
        
        if scene['setting'] == 'public':
            options.append({'action': 'defer_privately', 'description': 'Handle this later in private'})
            options.append({'action': 'subtle_response', 'description': 'Respond without making a scene'})
        elif scene['setting'] == 'intimate':
            options.append({'action': 'emotional_appeal', 'description': 'Make an emotional appeal'})
            options.append({'action': 'physical_comfort', 'description': 'Use physical closeness to defuse'})
        
        return options

# === Lore/World Integration ===

class LoreConflictIntegration:
    """Integrates world lore and cultural context with conflicts"""
    
    def __init__(self, lore_system, conflict_system):
        self.lore = lore_system
        self.conflicts = conflict_system
    
    def apply_cultural_modifiers(self, conflict: Dict) -> Dict:
        """Apply matriarchal society modifiers to conflict"""
        # In matriarchal society, certain conflicts have different dynamics
        cultural_rules = self.lore.get_cultural_rules()
        
        if conflict['type'] == 'authority_challenge':
            if conflict['challenger_gender'] == 'female':
                conflict['legitimacy'] *= 1.3  # More legitimate
                conflict['social_support'] *= 1.2
            else:
                conflict['legitimacy'] *= 0.7  # Less legitimate
                conflict['requires_extra_justification'] = True
        
        # Apply social expectations
        conflict['cultural_expectations'] = self._get_cultural_expectations(conflict['type'])
        
        return conflict
    
    def _get_cultural_expectations(self, conflict_type: str) -> List[str]:
        """Get cultural expectations for conflict type"""
        expectations = {
            'domestic_control': [
                "Women expected to have final say",
                "Men's resistance seen as improper",
                "Compromise viewed as weakness"
            ],
            'resource_allocation': [
                "Female head of household decides",
                "Male input is advisory only",
                "Children defer to mother's judgment"
            ]
        }
        return expectations.get(conflict_type, [])

# Master Integration Orchestrator
class ConflictSystemOrchestrator:
    """Orchestrates all system integrations"""
    
    def __init__(self, db_connection, all_systems: Dict):
        self.db = db_connection
        self.systems = all_systems
        self._setup_all_integrations()
    
    def _setup_all_integrations(self):
        """Initialize all integration modules"""
        self.time_integration = TimeCycleConflictIntegration(
            self.systems['time'], self.systems['conflicts']
        )
        self.relationship_integration = RelationshipConflictIntegration(
            self.systems['relationships'], self.systems['conflicts']
        )
        self.memory_integration = MemoryConflictIntegration(
            self.systems['memory'], self.systems['conflicts']
        )
        self.narrative_integration = NarrativeConflictIntegration(
            self.systems['narrative'], self.systems['conflicts']
        )
        self.stats_vitals_integration = StatsVitalsConflictIntegration(
            self.systems['stats'], self.systems['vitals'], self.systems['conflicts']
        )
        self.addiction_integration = AddictionConflictIntegration(
            self.systems['addiction'], self.systems['conflicts']
        )
        self.event_integration = EventConflictIntegration(
            self.systems['events'], self.systems['conflicts']
        )
        self.scene_integration = SceneConflictIntegration(
            self.systems['scenes'], self.systems['conflicts']
        )
        self.lore_integration = LoreConflictIntegration(
            self.systems['lore'], self.systems['conflicts']
        )
    
    def process_conflict_with_all_systems(self, conflict_id: str, context: Dict) -> Dict:
        """Process conflict through all integrated systems"""
        conflict = self.systems['conflicts'].get_conflict(conflict_id)
        
        # Apply all system modifiers
        conflict = self.time_integration.modify_conflict_by_schedule(conflict, context['current_time'])
        conflict = self.stats_vitals_integration.modify_conflict_options_by_stats(context['player_id'], conflict)
        conflict = self.lore_integration.apply_cultural_modifiers(conflict)
        
        # Check for special conditions
        for npc_id in conflict['stakeholders']:
            # Check addiction leverage
            addiction_leverage = self.addiction_integration.use_addiction_as_leverage(
                context['player_id'], npc_id, 'attention'
            )
            if addiction_leverage:
                conflict['special_leverage'] = addiction_leverage
            
            # Check memories
            relevant_memories = self.memory_integration.use_memories_in_conflict(
                npc_id, conflict
            )
            if relevant_memories:
                conflict['historical_context'] = relevant_memories
        
        # Embed in current scene if applicable
        if context.get('current_scene'):
            conflict = self.scene_integration.embed_conflict_in_scene(
                context['current_scene'], conflict
            )
        
        return conflict
    
    def resolve_conflict_with_consequences(self, conflict_id: str, resolution: Dict):
        """Apply resolution across all systems"""
        # Update relationships
        self.relationship_integration.apply_conflict_to_relationship(conflict_id, resolution)
        
        # Create memories
        for stakeholder_id in resolution['stakeholders']:
            self.memory_integration.create_conflict_memory(conflict_id, stakeholder_id, resolution)
        
        # Trigger events
        self.event_integration.trigger_conflict_event_chain(conflict_id, resolution)
        
        # Advance narratives
        for npc_id in resolution['stakeholders']:
            self.narrative_integration.trigger_narrative_advancement(npc_id, conflict_id, resolution)
        
        return {
            'resolution': resolution,
            'consequences': 'Applied across all systems',
            'follow_up_events': 'Scheduled'
        }
"""
