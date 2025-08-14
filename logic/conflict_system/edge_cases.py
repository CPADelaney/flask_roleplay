# Testing Framework & Edge Case Handling for Conflict System

import unittest
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import random

class EdgeCaseHandler:
    """Handles edge cases in the conflict system"""
    
    def __init__(self, conflict_system, db_connection):
        self.conflicts = conflict_system
        self.db = db_connection
    
    def handle_all_npcs_allied(self, player_id: str) -> Dict:
        """Handle case where all NPCs are in same alliance"""
        # Detect if all NPCs are allied
        all_npcs = self._get_all_npcs()
        alliances = self._get_npc_alliances(all_npcs)
        
        if len(set(alliances.values())) == 1:
            # All in same alliance - create internal friction
            return self._create_alliance_internal_conflict(list(alliances.keys()))
        return None
    
    def _create_alliance_internal_conflict(self, allied_npcs: List[str]) -> Dict:
        """Create conflict within an alliance"""
        # Pick two NPCs with highest tension
        tension_pairs = []
        for i, npc1 in enumerate(allied_npcs):
            for npc2 in allied_npcs[i+1:]:
                tension = self._calculate_tension(npc1, npc2)
                tension_pairs.append((tension, npc1, npc2))
        
        tension_pairs.sort(reverse=True)
        if tension_pairs and tension_pairs[0][0] > 0.3:
            _, npc1, npc2 = tension_pairs[0]
            return {
                'type': 'alliance_friction',
                'description': f"Tension within the alliance between {npc1} and {npc2}",
                'stakes': {
                    'alliance_stability': 'May split the alliance',
                    'hierarchy': 'Determine pecking order within alliance'
                }
            }
        return None
    
    def handle_player_disengagement(self, player_id: str) -> Dict:
        """Handle when player completely disengages from conflicts"""
        recent_responses = self._get_recent_conflict_responses(player_id, days=7)
        
        if len(recent_responses) == 0 or all(r['type'] == 'ignore' for r in recent_responses):
            # Player is disengaged - NPCs act autonomously
            return self._trigger_autonomous_resolution(player_id)
    
    def _trigger_autonomous_resolution(self, player_id: str) -> Dict:
        """NPCs resolve conflicts without player input"""
        active_conflicts = self.conflicts.get_active_conflicts()
        
        resolutions = []
        for conflict in active_conflicts:
            if self._should_auto_resolve(conflict):
                resolution = self._generate_autonomous_resolution(conflict)
                resolutions.append(resolution)
                
                # Apply resolution
                self.conflicts.resolve_conflict(conflict['id'], resolution)
        
        return {
            'auto_resolved': len(resolutions),
            'message': "NPCs resolved conflicts without your input",
            'consequences': self._summarize_consequences(resolutions)
        }
    
    def handle_conflicting_background_events(self, events: List[Dict]) -> List[Dict]:
        """Resolve conflicting background events"""
        # Group events by type and location
        event_groups = {}
        for event in events:
            key = (event['type'], event.get('location', 'global'))
            if key not in event_groups:
                event_groups[key] = []
            event_groups[key].append(event)
        
        # Resolve conflicts within groups
        resolved_events = []
        for group_key, group_events in event_groups.items():
            if len(group_events) > 1:
                # Events conflict - merge or prioritize
                resolved = self._merge_conflicting_events(group_events)
                resolved_events.append(resolved)
            else:
                resolved_events.extend(group_events)
        
        return resolved_events
    
    def handle_memory_overload(self, npc_id: str) -> Dict:
        """Handle when memory system has too many conflict memories"""
        memory_count = self._count_conflict_memories(npc_id)
        
        if memory_count > 100:  # Threshold for overload
            # Consolidate memories into patterns
            patterns = self._consolidate_memories_to_patterns(npc_id)
            
            # Archive old detailed memories
            self._archive_old_memories(npc_id, days=30)
            
            return {
                'action': 'memory_consolidation',
                'patterns_extracted': len(patterns),
                'memories_archived': memory_count - 20,
                'message': f"Consolidated {npc_id}'s conflict memories into behavioral patterns"
            }
        return None
    
    def handle_circular_alliance_dependencies(self) -> Dict:
        """Detect and resolve circular alliance dependencies"""
        alliances = self._get_all_alliances()
        cycles = self._detect_cycles(alliances)
        
        if cycles:
            # Break cycles by weakening weakest links
            for cycle in cycles:
                weakest_link = self._find_weakest_alliance_link(cycle)
                self._weaken_alliance(weakest_link['from'], weakest_link['to'])
            
            return {
                'cycles_detected': len(cycles),
                'cycles_broken': len(cycles),
                'method': 'Weakened unstable alliance links'
            }
        return None
    
    def handle_resource_deadlock(self) -> Dict:
        """Handle when resources are completely locked/unavailable"""
        resources = self._get_all_resources()
        deadlocked = []
        
        for resource in resources:
            if self._is_deadlocked(resource):
                deadlocked.append(resource)
        
        if deadlocked:
            # Force redistribution
            for resource in deadlocked:
                self._force_redistribute(resource)
            
            return {
                'deadlocked_resources': len(deadlocked),
                'action': 'forced_redistribution',
                'message': 'Resources redistributed to break deadlock'
            }
        return None
    
    def handle_investigation_dead_end(self, investigation_id: str) -> Dict:
        """Handle when investigation reaches dead end"""
        investigation = self._get_investigation(investigation_id)
        
        if investigation['progress'] < 0.3 and investigation['days_active'] > 7:
            # Investigation stalled - provide hint or alternate path
            hint = self._generate_investigation_hint(investigation)
            alternate_path = self._create_alternate_investigation_path(investigation)
            
            return {
                'action': 'investigation_assistance',
                'hint': hint,
                'alternate_path': alternate_path,
                'message': 'New lead discovered in investigation'
            }
        return None
    
    def handle_canon_contradiction(self, fact1: Dict, fact2: Dict) -> Dict:
        """Resolve contradicting canonical facts"""
        # Determine which fact takes precedence
        if fact1['establishment_date'] < fact2['establishment_date']:
            # Earlier fact has precedence
            primary_fact = fact1
            contradicting_fact = fact2
        else:
            primary_fact = fact2
            contradicting_fact = fact1
        
        # Check if facts can coexist with modification
        if self._can_reconcile_facts(primary_fact, contradicting_fact):
            # Modify newer fact to be compatible
            modified_fact = self._reconcile_facts(primary_fact, contradicting_fact)
            return {
                'resolution': 'reconciled',
                'primary_fact': primary_fact,
                'modified_fact': modified_fact
            }
        else:
            # Newer fact is rejected
            self._reject_canonical_fact(contradicting_fact['id'])
            return {
                'resolution': 'rejected',
                'kept_fact': primary_fact,
                'rejected_fact': contradicting_fact,
                'reason': 'Contradicts established canon'
            }

class ConflictSystemTester:
    """Comprehensive testing for conflict system"""
    
    def __init__(self, conflict_system):
        self.conflicts = conflict_system
        self.test_results = []
    
    def run_all_tests(self) -> Dict:
        """Run comprehensive test suite"""
        tests = [
            self.test_conflict_frequency,
            self.test_resolution_timing,
            self.test_player_agency,
            self.test_multi_day_progression,
            self.test_alliance_stability,
            self.test_background_pacing,
            self.test_pattern_detection,
            self.test_resource_scarcity,
            self.test_social_reputation
        ]
        
        for test in tests:
            result = test()
            self.test_results.append(result)
        
        return self._summarize_test_results()
    
    def test_conflict_frequency(self) -> Dict:
        """Test that 3-5 conflicts generate naturally per week"""
        # Simulate a week
        conflicts_generated = []
        for day in range(7):
            daily_conflicts = self._simulate_day_conflicts()
            conflicts_generated.extend(daily_conflicts)
        
        count = len(conflicts_generated)
        passed = 3 <= count <= 5
        
        return {
            'test': 'conflict_frequency',
            'passed': passed,
            'expected': '3-5 per week',
            'actual': count,
            'conflicts': conflicts_generated
        }
    
    def test_resolution_timing(self) -> Dict:
        """Test that conflicts resolve in 5-20 days through patterns"""
        test_conflicts = self._create_test_conflicts(5)
        resolution_times = []
        
        for conflict in test_conflicts:
            days_to_resolve = self._simulate_resolution(conflict)
            resolution_times.append(days_to_resolve)
        
        avg_time = sum(resolution_times) / len(resolution_times)
        passed = 5 <= avg_time <= 20
        
        return {
            'test': 'resolution_timing',
            'passed': passed,
            'expected': '5-20 days average',
            'actual': f'{avg_time:.1f} days average',
            'times': resolution_times
        }
    
    def test_player_agency(self) -> Dict:
        """Test that player can ignore 90% of conflicts without breaking game"""
        conflicts = self._create_test_conflicts(20)
        ignored = random.sample(conflicts, 18)  # Ignore 90%
        engaged = [c for c in conflicts if c not in ignored]
        
        # Simulate ignoring conflicts
        game_state_before = self._capture_game_state()
        for conflict in ignored:
            self._simulate_ignore_conflict(conflict)
        game_state_after = self._capture_game_state()
        
        # Check game still functions
        game_functional = self._check_game_functionality(game_state_after)
        meaningful_progress = self._check_meaningful_progress(engaged)
        
        return {
            'test': 'player_agency',
            'passed': game_functional and meaningful_progress,
            'ignored_percentage': 90,
            'game_functional': game_functional,
            'meaningful_progress': meaningful_progress
        }
    
    def test_multi_day_progression(self) -> Dict:
        """Test conflict progression over multiple days"""
        conflict = self._create_test_conflict()
        progression_log = []
        
        for day in range(10):
            daily_progress = self._simulate_daily_progression(conflict, day)
            progression_log.append(daily_progress)
        
        # Check for natural progression
        has_escalation = any(p['intensity_change'] > 0 for p in progression_log)
        has_deescalation = any(p['intensity_change'] < 0 for p in progression_log)
        has_stasis = any(p['intensity_change'] == 0 for p in progression_log)
        
        natural_progression = has_escalation and has_deescalation and has_stasis
        
        return {
            'test': 'multi_day_progression',
            'passed': natural_progression,
            'progression': progression_log,
            'has_variety': natural_progression
        }
    
    def test_alliance_stability(self) -> Dict:
        """Test alliance stability over time"""
        alliances = self._create_test_alliances(3)
        stability_log = []
        
        for day in range(14):
            daily_stability = self._measure_alliance_stability(alliances)
            stability_log.append(daily_stability)
            
            # Apply daily pressures
            self._apply_alliance_pressures(alliances)
        
        # Check stability patterns
        avg_stability = sum(s['overall'] for s in stability_log) / len(stability_log)
        has_changes = len(set(s['overall'] for s in stability_log)) > 1
        
        return {
            'test': 'alliance_stability',
            'passed': 0.3 <= avg_stability <= 0.8 and has_changes,
            'average_stability': avg_stability,
            'dynamic': has_changes
        }
    
    def test_background_pacing(self) -> Dict:
        """Test pacing of background conflicts"""
        background_events = []
        
        for hour in range(168):  # One week in hours
            events = self._generate_background_events(hour)
            background_events.extend(events)
        
        # Analyze pacing
        events_per_day = [0] * 7
        for event in background_events:
            day = event['hour'] // 24
            events_per_day[day] += 1
        
        avg_per_day = sum(events_per_day) / 7
        variation = max(events_per_day) - min(events_per_day)
        
        good_pacing = 2 <= avg_per_day <= 5 and variation <= 3
        
        return {
            'test': 'background_pacing',
            'passed': good_pacing,
            'events_per_day': events_per_day,
            'average': avg_per_day,
            'variation': variation
        }
    
    def test_pattern_detection(self) -> Dict:
        """Test pattern detection thresholds"""
        patterns = [
            {'type': 'morning_control', 'instances': 3, 'detected': False},
            {'type': 'attention_seeking', 'instances': 5, 'detected': True},
            {'type': 'resource_hoarding', 'instances': 2, 'detected': False}
        ]
        
        correct_detections = 0
        for pattern in patterns:
            detected = self._test_pattern_detection(pattern)
            if detected == pattern['detected']:
                correct_detections += 1
        
        accuracy = correct_detections / len(patterns)
        
        return {
            'test': 'pattern_detection',
            'passed': accuracy >= 0.8,
            'accuracy': accuracy,
            'patterns_tested': len(patterns)
        }
    
    def test_resource_scarcity(self) -> Dict:
        """Test resource scarcity balance"""
        resources = self._create_test_resources(5)
        scarcity_events = []
        
        for day in range(7):
            daily_scarcity = self._simulate_resource_scarcity(resources)
            scarcity_events.append(daily_scarcity)
        
        # Check balance
        total_scarcity = sum(e['scarcity_level'] for e in scarcity_events)
        avg_scarcity = total_scarcity / len(scarcity_events)
        
        # Should create tension but not constant crisis
        balanced = 0.2 <= avg_scarcity <= 0.5
        
        return {
            'test': 'resource_scarcity',
            'passed': balanced,
            'average_scarcity': avg_scarcity,
            'target_range': '0.2-0.5'
        }
    
    def test_social_reputation(self) -> Dict:
        """Test social reputation impact rates"""
        test_npc = self._create_test_npc()
        reputation_changes = []
        
        # Simulate various actions
        actions = [
            {'type': 'conflict_win', 'impact': 0.1},
            {'type': 'conflict_loss', 'impact': -0.1},
            {'type': 'compromise', 'impact': 0.05},
            {'type': 'betrayal', 'impact': -0.3}
        ]
        
        for action in actions:
            impact = self._apply_reputation_change(test_npc, action)
            reputation_changes.append(impact)
        
        # Check if impacts are proportional and reasonable
        reasonable_impacts = all(-0.5 <= i <= 0.5 for i in reputation_changes)
        
        return {
            'test': 'social_reputation',
            'passed': reasonable_impacts,
            'reputation_changes': reputation_changes,
            'reasonable': reasonable_impacts
        }
    
    def _summarize_test_results(self) -> Dict:
        """Summarize all test results"""
        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total,
            'details': self.test_results,
            'ready_for_production': passed / total >= 0.8
        }

class BalanceAdjuster:
    """Adjusts system balance based on testing"""
    
    def __init__(self, conflict_system):
        self.conflicts = conflict_system
        self.balance_params = self._load_balance_params()
    
    def _load_balance_params(self) -> Dict:
        """Load current balance parameters"""
        return {
            'conflict_frequency': 0.3,
            'intensity_escalation': 0.1,
            'resolution_difficulty': 0.5,
            'stakeholder_action_rate': 0.4,
            'leverage_decay': 0.05,
            'canon_strength': 0.7,
            'discovery_chance': 0.2
        }
    
    def auto_balance(self, test_results: Dict):
        """Automatically adjust balance based on test results"""
        adjustments = []
        
        # Adjust conflict frequency
        if test_results['conflict_frequency']['actual'] < 3:
            self.balance_params['conflict_frequency'] *= 1.2
            adjustments.append('Increased conflict frequency')
        elif test_results['conflict_frequency']['actual'] > 5:
            self.balance_params['conflict_frequency'] *= 0.8
            adjustments.append('Decreased conflict frequency')
        
        # Adjust resolution difficulty
        if test_results['resolution_timing']['actual'] < 5:
            self.balance_params['resolution_difficulty'] *= 1.3
            adjustments.append('Increased resolution difficulty')
        elif test_results['resolution_timing']['actual'] > 20:
            self.balance_params['resolution_difficulty'] *= 0.7
            adjustments.append('Decreased resolution difficulty')
        
        self._save_balance_params()
        
        return {
            'adjustments_made': len(adjustments),
            'adjustments': adjustments,
            'new_params': self.balance_params
        }
    
    def tune_parameter(self, param_name: str, new_value: float) -> Dict:
        """Manually tune a specific parameter"""
        if param_name not in self.balance_params:
            return {'error': f'Unknown parameter: {param_name}'}
        
        old_value = self.balance_params[param_name]
        self.balance_params[param_name] = new_value
        self._save_balance_params()
        
        return {
            'parameter': param_name,
            'old_value': old_value,
            'new_value': new_value,
            'change_percentage': ((new_value - old_value) / old_value) * 100
        }
    
    def _save_balance_params(self):
        """Save balance parameters to database"""
        # Implementation would save to database
        pass

# Performance Optimizer
class ConflictPerformanceOptimizer:
    """Optimizes conflict system performance"""
    
    def __init__(self, conflict_system, db_connection):
        self.conflicts = conflict_system
        self.db = db_connection
    
    def optimize_queries(self):
        """Optimize database queries"""
        # Create indexes for frequently accessed data
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_conflicts_active ON conflicts(active)",
            "CREATE INDEX IF NOT EXISTS idx_tensions_player ON tension_accumulation(player_id)",
            "CREATE INDEX IF NOT EXISTS idx_precedents_date ON precedents(established_date)",
            "CREATE INDEX IF NOT EXISTS idx_leverage_expiry ON social_leverage(expires_at)"
        ]
        
        for index in indexes:
            self.db.execute(index)
        
        return {'indexes_created': len(indexes)}
    
    def cache_frequent_calculations(self):
        """Cache frequently calculated values"""
        cache_config = {
            'tension_levels': {'ttl': 300},  # 5 minutes
            'alliance_status': {'ttl': 600},  # 10 minutes
            'dominance_scores': {'ttl': 900},  # 15 minutes
            'pattern_detections': {'ttl': 1800}  # 30 minutes
        }
        
        # Implementation would set up caching
        return {'caches_configured': len(cache_config)}
    
    def batch_operations(self):
        """Batch database operations for efficiency"""
        # Implementation would batch inserts/updates
        return {'batching': 'enabled'}

# Main test runner
def run_complete_system_test():
    """Run complete system test and optimization"""
    # Initialize systems
    conflict_system = None  # Would be actual system instance
    db_connection = None  # Would be actual database connection
    
    # Run tests
    tester = ConflictSystemTester(conflict_system)
    test_results = tester.run_all_tests()
    
    # Handle edge cases
    edge_handler = EdgeCaseHandler(conflict_system, db_connection)
    edge_cases_handled = {
        'all_npcs_allied': edge_handler.handle_all_npcs_allied('player_1'),
        'player_disengaged': edge_handler.handle_player_disengagement('player_1'),
        'memory_overload': edge_handler.handle_memory_overload('npc_1'),
        'circular_alliances': edge_handler.handle_circular_alliance_dependencies(),
        'resource_deadlock': edge_handler.handle_resource_deadlock()
    }
    
    # Auto-balance
    adjuster = BalanceAdjuster(conflict_system)
    balance_adjustments = adjuster.auto_balance(test_results)
    
    # Optimize performance
    optimizer = ConflictPerformanceOptimizer(conflict_system, db_connection)
    optimizations = {
        'queries': optimizer.optimize_queries(),
        'caching': optimizer.cache_frequent_calculations(),
        'batching': optimizer.batch_operations()
    }
    
    return {
        'test_results': test_results,
        'edge_cases': edge_cases_handled,
        'balance': balance_adjustments,
        'optimizations': optimizations,
        'system_ready': test_results['ready_for_production']
    }
"""
