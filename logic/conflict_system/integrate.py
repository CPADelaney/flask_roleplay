# Integrated Conflict System - Game Flow Implementation

from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import random

class IntegratedConflictManager:
    """Main manager that orchestrates all conflict subsystems"""
    
    def __init__(self, game_state):
        # Core systems
        self.tension_analyzer = TensionAnalyzer()
        self.object_generator = ObjectGenerator()
        self.leverage_system = LeverageSystem()
        
        # Game state references
        self.game_state = game_state
        self.time_cycle = game_state.time_cycle
        self.relationship_system = game_state.relationships
        self.memory_system = game_state.memories
        self.npc_system = game_state.npcs
        self.event_system = game_state.events
        
        # Object and leverage registries
        self.meaningful_objects: Dict[str, MeaningfulObject] = {}
        self.active_conflicts: List[Dict] = []
        
        # Configuration
        self.config = {
            'tension_check_frequency': 60,  # Check every game hour
            'max_concurrent_conflicts': 3,
            'background_conflict_chance': 0.3,
            'micro_aggression_frequency': 0.2
        }
    
    def hourly_update(self, current_time: datetime):
        """Called every game hour to update all systems"""
        
        # 1. Apply time decay to tensions
        self._apply_tension_decay(1.0)
        
        # 2. Check for brewing tensions
        brewing = self._check_all_brewing_tensions()
        
        # 3. Process autonomous NPC actions
        npc_actions = self._process_npc_autonomy()
        
        # 4. Update leverage decay
        self._update_leverage_decay()
        
        # 5. Check for conflict emergence
        emergent_conflicts = self._check_conflict_emergence(brewing)
        
        # 6. Process background conflicts
        self._update_background_conflicts()
        
        return {
            'brewing_tensions': brewing,
            'npc_actions': npc_actions,
            'emerged_conflicts': emergent_conflicts,
            'active_conflicts': len(self.active_conflicts)
        }
    
    def _apply_tension_decay(self, hours: float):
        """Apply natural tension decay over time"""
        for accumulator in self.tension_analyzer.accumulators.values():
            accumulator.apply_time_decay(hours)
            
            # Weekend modifier - tensions decay faster
            if self.time_cycle.is_weekend():
                accumulator.apply_time_decay(hours * 0.2)  # Extra 20% decay
    
    def _check_all_brewing_tensions(self) -> List[Dict]:
        """Check all character pairs for brewing tensions"""
        brewing = []
        scene_context = self._get_current_scene_context()
        
        # Get all characters in current location
        present_characters = self.npc_system.get_present_characters(
            self.game_state.current_location
        )
        
        if len(present_characters) >= 2:
            # Check group dynamics
            tension_web = self.tension_analyzer.calculate_group_tension_dynamics(
                present_characters, 
                scene_context
            )
            
            for (char_a, char_b), tension_data in tension_web.items():
                if tension_data['pressure'] > 5:  # Noteworthy tension
                    brewing.append({
                        'participants': [char_a, char_b],
                        'pressure': tension_data['pressure'],
                        'type': tension_data.get('dominant_source'),
                        'suggested_expression': self._suggest_tension_expression(
                            tension_data
                        )
                    })
        
        return brewing
    
    def _suggest_tension_expression(self, tension_data: Dict) -> str:
        """Suggest how tension manifests in scene"""
        intensity = tension_data.get('intensity_level', 'subtext')
        
        expressions = {
            'subtext': [
                "slight pause before responding",
                "avoiding direct eye contact",
                "overly polite tone",
                "subtle change in body language"
            ],
            'tension': [
                "pointed silence",
                "clipped responses",
                "passive aggressive comment",
                "obvious discomfort"
            ],
            'passive_aggressive': [
                "backhanded compliment",
                "bringing up old grievances",
                "deliberate misunderstanding",
                "exaggerated compliance"
            ],
            'confrontation': [
                "direct challenge",
                "raised voice",
                "accusation",
                "ultimatum"
            ]
        }
        
        options = expressions.get(intensity, expressions['subtext'])
        return random.choice(options)
    
    def _process_npc_autonomy(self) -> List[Dict]:
        """Process autonomous NPC conflict actions"""
        actions = []
        
        for npc_id, npc_data in self.npc_system.get_all_npcs().items():
            # Check if NPC should take autonomous action
            if self._should_npc_act_autonomously(npc_id, npc_data):
                action = self._generate_autonomous_action(npc_id, npc_data)
                if action:
                    actions.append(action)
                    self._apply_autonomous_action(action)
        
        return actions
    
    def _should_npc_act_autonomously(self, npc_id: str, npc_data: Dict) -> bool:
        """Determine if NPC should take autonomous action"""
        # Based on personality, current tensions, and random chance
        autonomy_score = npc_data.get('autonomy', 0.5)
        
        # Check if involved in tensions
        has_tension = any(
            npc_id in [acc.character_a, acc.character_b] 
            for acc in self.tension_analyzer.accumulators.values()
            if acc.current_pressure > 3
        )
        
        if has_tension:
            autonomy_score *= 1.5
            
        return random.random() < autonomy_score * 0.1  # 10% base chance
    
    def _generate_autonomous_action(self, npc_id: str, npc_data: Dict) -> Optional[Dict]:
        """Generate autonomous conflict action for NPC"""
        action_types = [
            'establish_precedent',
            'reinforce_pattern',
            'subtle_rebellion',
            'claim_object',
            'build_alliance',
            'micro_aggression'
        ]
        
        action_type = random.choice(action_types)
        
        if action_type == 'establish_precedent':
            return {
                'actor': npc_id,
                'type': 'establish_precedent',
                'precedent': self._generate_precedent(npc_id),
                'visibility': 'subtle'  # Most autonomous actions are subtle
            }
        elif action_type == 'claim_object':
            # Try to claim an unclaimed meaningful object
            for obj_id, obj in self.meaningful_objects.items():
                if obj.owner is None and npc_id in obj.associated_characters:
                    return {
                        'actor': npc_id,
                        'type': 'claim_object',
                        'object': obj_id,
                        'method': 'quiet_assumption'
                    }
        elif action_type == 'micro_aggression':
            # Generate micro-aggression toward someone with tension
            for acc in self.tension_analyzer.accumulators.values():
                if npc_id == acc.character_a and acc.current_pressure > 2:
                    return {
                        'actor': npc_id,
                        'type': 'micro_aggression',
                        'target': acc.character_b,
                        'action': self._generate_micro_aggression(npc_id, acc.character_b)
                    }
        
        return None
    
    def _generate_precedent(self, npc_id: str) -> Dict:
        """Generate a precedent an NPC is trying to establish"""
        precedents = [
            {'type': 'seating', 'claim': 'always sits here'},
            {'type': 'timing', 'claim': 'gets first shower'},
            {'type': 'decision', 'claim': 'chooses what to watch'},
            {'type': 'territory', 'claim': 'this is their space'},
            {'type': 'routine', 'claim': 'this is how we do things'}
        ]
        return random.choice(precedents)
    
    def _generate_micro_aggression(self, actor: str, target: str) -> str:
        """Generate contextual micro-aggression"""
        aggressions = [
            "interrupts mid-sentence",
            "doesn't acknowledge greeting",
            "makes dismissive sound",
            "turns away while talking",
            "checks phone while listening",
            "gives minimal response",
            "uses condescending tone"
        ]
        return random.choice(aggressions)
    
    def _apply_autonomous_action(self, action: Dict):
        """Apply effects of autonomous NPC action"""
        if action['type'] == 'establish_precedent':
            # Record in memory system
            self.memory_system.add_memory({
                'type': 'precedent_attempt',
                'actor': action['actor'],
                'precedent': action['precedent'],
                'timestamp': datetime.now()
            })
            
        elif action['type'] == 'claim_object':
            # Update object ownership
            if action['object'] in self.meaningful_objects:
                self.meaningful_objects[action['object']].owner = action['actor']
                
        elif action['type'] == 'micro_aggression':
            # Add to tension accumulator
            key = tuple(sorted([action['actor'], action['target']]))
            if key in self.tension_analyzer.accumulators:
                self.tension_analyzer.accumulators[key].add_micro_aggression({
                    'action': action['action'],
                    'weight': 0.15
                })
    
    def _check_conflict_emergence(self, brewing_tensions: List[Dict]) -> List[Dict]:
        """Check if any tensions should emerge as active conflicts"""
        emerged = []
        
        for tension in brewing_tensions:
            if tension['pressure'] >= 10 and len(self.active_conflicts) < self.config['max_concurrent_conflicts']:
                # Check if there's a trigger in current scene
                trigger = self._find_conflict_trigger(tension)
                
                if trigger:
                    conflict = self._create_active_conflict(tension, trigger)
                    self.active_conflicts.append(conflict)
                    emerged.append(conflict)
                    
                    # Reset pressure after emergence
                    key = tuple(sorted(tension['participants']))
                    if key in self.tension_analyzer.accumulators:
                        self.tension_analyzer.accumulators[key].current_pressure *= 0.3
        
        return emerged
    
    def _find_conflict_trigger(self, tension: Dict) -> Optional[Dict]:
        """Find appropriate trigger for conflict emergence"""
        scene_context = self._get_current_scene_context()
        
        # Check for object-based triggers
        for obj_id, obj in self.meaningful_objects.items():
            if set(tension['participants']) <= obj.associated_characters:
                if obj.is_contested or random.random() < 0.3:
                    return {
                        'type': 'object_dispute',
                        'object': obj_id,
                        'reason': 'contested_ownership'
                    }
        
        # Check for routine-based triggers
        if scene_context.get('routine_moment'):
            return {
                'type': 'routine_disruption',
                'routine': scene_context['routine_moment'],
                'reason': 'pattern_violation'
            }
        
        # Default trigger based on tension type
        if tension.get('type'):
            return {
                'type': 'accumulated_tension',
                'source': tension['type'],
                'reason': 'boiling_point'
            }
        
        return None
    
    def _create_active_conflict(self, tension: Dict, trigger: Dict) -> Dict:
        """Create an active conflict from tension and trigger"""
        return {
            'id': f"conflict_{datetime.now().timestamp()}",
            'participants': tension['participants'],
            'intensity': self._calculate_intensity(tension['pressure']),
            'trigger': trigger,
            'stage': 'emerging',
            'resolution_progress': 0.0,
            'pattern_count': 0,
            'created_at': datetime.now(),
            'background': False  # Player-visible conflict
        }
    
    def _calculate_intensity(self, pressure: float) -> str:
        """Calculate conflict intensity from pressure"""
        if pressure < 12:
            return 'subtext'
        elif pressure < 15:
            return 'tension'
        elif pressure < 20:
            return 'argument'
        else:
            return 'confrontation'
    
    def _update_background_conflicts(self):
        """Update conflicts happening in background"""
        if random.random() < self.config['background_conflict_chance']:
            # Generate background conflict between NPCs not in scene
            absent_npcs = self.npc_system.get_absent_characters(
                self.game_state.current_location
            )
            
            if len(absent_npcs) >= 2:
                participants = random.sample(absent_npcs, 2)
                self.active_conflicts.append({
                    'id': f"bg_conflict_{datetime.now().timestamp()}",
                    'participants': participants,
                    'intensity': 'subtext',
                    'stage': 'ongoing',
                    'background': True,
                    'player_visible': False
                })
    
    def _get_current_scene_context(self) -> Dict:
        """Get context of current scene for conflict processing"""
        return {
            'location': self.game_state.current_location,
            'time_of_day': self.time_cycle.get_period(),
            'day_type': 'weekend' if self.time_cycle.is_weekend() else 'weekday',
            'present_npcs': self.npc_system.get_present_characters(
                self.game_state.current_location
            ),
            'player_present': True,
            'routine_moment': self._check_routine_moment(),
            'shared_resources': self._get_shared_resources(),
            'hungry': any(self.game_state.vitals.get(npc, {}).get('hunger', 0) > 0.7 
                         for npc in self.npc_system.get_present_characters(
                             self.game_state.current_location)),
            'tired': self.time_cycle.get_period() in ['late_evening', 'night']
        }
    
    def _check_routine_moment(self) -> Optional[str]:
        """Check if current time matches a routine moment"""
        routines = {
            'morning': 'breakfast',
            'evening': 'dinner',
            'night': 'bedtime',
            'afternoon': 'leisure'
        }
        return routines.get(self.time_cycle.get_period())
    
    def _get_shared_resources(self) -> List[Dict]:
        """Get resources that might be contested"""
        location = self.game_state.current_location
        resources = []
        
        if location == 'bathroom':
            resources.append({
                'name': 'hot water',
                'availability': 0.7,
                'demand': 1.0
            })
        elif location == 'living_room':
            resources.append({
                'name': 'TV control',
                'availability': 1.0,
                'demand': len(self.npc_system.get_present_characters(location)) * 0.5
            })
            
        return resources
    
    def _update_leverage_decay(self):
        """Update leverage decay daily"""
        if self.time_cycle.is_new_day():
            for leverages in self.leverage_system.leverage_map.values():
                for leverage in leverages:
                    leverage.decay(1)

# LLM Agent Integration Functions

def get_conflict_state(manager: IntegratedConflictManager) -> str:
    """LLM tool to get current conflict state"""
    state = {
        'active_conflicts': len(manager.active_conflicts),
        'brewing_tensions': [],
        'available_objects': [],
        'leverage_dynamics': {}
    }
    
    # Get brewing tensions
    brewing = manager._check_all_brewing_tensions()
    for tension in brewing[:3]:  # Top 3 tensions
        state['brewing_tensions'].append({
            'participants': tension['participants'],
            'pressure': tension['pressure'],
            'expression': tension.get('suggested_expression')
        })
    
    # Get meaningful objects in scene
    location = manager.game_state.current_location
    for obj_id, obj in manager.meaningful_objects.items():
        if obj.location == location:
            state['available_objects'].append({
                'name': obj.name,
                'owner': obj.owner,
                'significance': obj.significance_type.value,
                'contested': obj.is_contested
            })
    
    # Get leverage for present characters
    present = manager.npc_system.get_present_characters(location)
    for char in present:
        if char != 'player':
            leverage = manager.leverage_system.calculate_total_leverage(
                'player', char
            )
            state['leverage_dynamics'][char] = leverage
    
    return json.dumps(state, indent=2)

def suggest_conflict_action(manager: IntegratedConflictManager, 
                           conflict_id: str) -> str:
    """LLM tool to suggest player actions in conflict"""
    conflict = None
    for c in manager.active_conflicts:
        if c['id'] == conflict_id:
            conflict = c
            break
    
    if not conflict:
        return json.dumps({'error': 'Conflict not found'})
    
    suggestions = []
    
    # Based on intensity
    if conflict['intensity'] == 'subtext':
        suggestions.extend([
            "Acknowledge tension without addressing directly",
            "Make subtle counter-move",
            "Redirect conversation",
            "Use humor to defuse"
        ])
    elif conflict['intensity'] == 'tension':
        suggestions.extend([
            "Address issue directly but calmly",
            "Establish boundary",
            "Seek compromise",
            "Assert position firmly"
        ])
    
    # Check for leverage options
    for participant in conflict['participants']:
        if participant != 'player':
            leverage = manager.leverage_system.calculate_total_leverage(
                'player', participant
            )
            if leverage > 0.3:
                suggestions.append(f"Use leverage over {participant}")
    
    # Check for object options
    if conflict['trigger'].get('type') == 'object_dispute':
        obj_id = conflict['trigger']['object']
        if obj_id in manager.meaningful_objects:
            suggestions.append("Negotiate shared use of object")
            suggestions.append("Trade object for concession")
    
    return json.dumps({
        'conflict_id': conflict_id,
        'intensity': conflict['intensity'],
        'suggestions': suggestions,
        'recommendation': suggestions[0] if suggestions else "Observe and wait"
    })

# Example Usage in Game Loop

def game_hour_tick(game_state, conflict_manager: IntegratedConflictManager):
    """Called every game hour"""
    current_time = game_state.time_cycle.current_time
    
    # Update conflict systems
    update_result = conflict_manager.hourly_update(current_time)
    
    # Check for notable events to surface to player
    notifications = []
    
    # Brewing tensions in current scene
    if update_result['brewing_tensions']:
        for tension in update_result['brewing_tensions']:
            if 'player' in tension['participants'] or tension['pressure'] > 8:
                notifications.append({
                    'type': 'tension',
                    'message': f"You notice {tension['suggested_expression']} between {' and '.join(tension['participants'])}"
                })
    
    # Emerged conflicts
    for conflict in update_result['emerged_conflicts']:
        if 'player' in conflict['participants']:
            notifications.append({
                'type': 'conflict_emerged',
                'message': f"Tension boils over about {conflict['trigger']['reason']}"
            })
    
    # Autonomous NPC actions witnessed
    for action in update_result['npc_actions']:
        if action.get('visibility') != 'hidden':
            notifications.append({
                'type': 'npc_action',
                'message': f"{action['actor']} {action.get('action', 'acts autonomously')}"
            })
    
    return notifications
