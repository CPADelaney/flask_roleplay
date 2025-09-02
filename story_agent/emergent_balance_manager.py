# story_agent/emergent_balance_manager.py

"""
Manages balance in the open-world simulation between:
- Player agency and NPC influence
- Routine and variety
- Subtle and overt power dynamics
- Emergent and guided experiences
"""

import random
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from story_agent.world_simulation_models import WorldState, WorldMood
from story_agent.emergent_story_tracker import EmergentNarrative, EmergentPattern

class BalanceAxis(Enum):
    """Different axes to balance in the simulation"""
    AGENCY_CONTROL = "agency_control"  # Player freedom vs NPC control
    ROUTINE_VARIETY = "routine_variety"  # Predictable vs surprising
    SUBTLE_OVERT = "subtle_overt"  # Hidden vs obvious power dynamics
    EMERGENT_GUIDED = "emergent_guided"  # Natural vs directed events

@dataclass
class SimulationBalance:
    """Current balance state of the simulation"""
    agency_level: float = 0.5  # 0 = full NPC control, 1 = full player agency
    routine_level: float = 0.5  # 0 = chaos, 1 = complete routine
    subtlety_level: float = 0.7  # 0 = overt control, 1 = completely hidden
    emergence_level: float = 0.7  # 0 = fully guided, 1 = fully emergent

class EmergentBalanceManager:
    """Manages balance in the open-world simulation"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.current_balance = SimulationBalance()
        self.balance_history: List[SimulationBalance] = []
        self.adjustment_cooldown = 0
        
    async def evaluate_current_balance(
        self,
        world_state: WorldState,
        recent_events: List[Dict[str, Any]],
        active_narratives: List[EmergentNarrative]
    ) -> SimulationBalance:
        """Evaluate the current balance of the simulation"""
        
        # Calculate agency level
        submission_level = world_state.relationship_dynamics.player_submission_level
        resistance_level = world_state.relationship_dynamics.resistance_level
        self.current_balance.agency_level = resistance_level / max(1, submission_level + resistance_level)
        
        # Calculate routine level
        routine_events = sum(1 for e in recent_events if e.get("type") == "routine")
        total_events = len(recent_events) or 1
        self.current_balance.routine_level = routine_events / total_events
        
        # Calculate subtlety level
        power_visibility = world_state.relationship_dynamics.power_visibility
        self.current_balance.subtlety_level = 1.0 - power_visibility
        
        # Calculate emergence level
        if active_narratives:
            planned_narratives = sum(1 for n in active_narratives 
                                   if n.pattern in [EmergentPattern.DAILY_CONDITIONING])
            emergence_ratio = 1.0 - (planned_narratives / len(active_narratives))
            self.current_balance.emergence_level = emergence_ratio
        
        # Store in history
        self.balance_history.append(self.current_balance)
        if len(self.balance_history) > 100:
            self.balance_history.pop(0)
        
        return self.current_balance
    
    async def suggest_balance_adjustments(
        self,
        target_balance: Optional[SimulationBalance] = None
    ) -> Dict[str, Any]:
        """Suggest adjustments to improve simulation balance"""
        
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return {"adjustments_needed": False}
        
        if not target_balance:
            # Default target balance for slice-of-life
            target_balance = SimulationBalance(
                agency_level=0.4,  # Some player agency but NPCs have influence
                routine_level=0.6,  # More routine than chaos
                subtlety_level=0.7,  # Mostly subtle power dynamics
                emergence_level=0.8  # Mostly emergent
            )
        
        adjustments = {}
        
        # Check agency balance
        agency_diff = abs(self.current_balance.agency_level - target_balance.agency_level)
        if agency_diff > 0.2:
            if self.current_balance.agency_level > target_balance.agency_level:
                adjustments["increase_npc_influence"] = True
                adjustments["suggested_dynamics"] = ["casual_dominance", "protective_control"]
            else:
                adjustments["increase_player_choices"] = True
                adjustments["reduce_forced_events"] = True
        
        # Check routine balance
        routine_diff = abs(self.current_balance.routine_level - target_balance.routine_level)
        if routine_diff > 0.2:
            if self.current_balance.routine_level > target_balance.routine_level:
                adjustments["add_random_encounters"] = True
                adjustments["vary_daily_activities"] = True
            else:
                adjustments["establish_routines"] = True
                adjustments["repeat_successful_patterns"] = True
        
        # Check subtlety balance
        subtlety_diff = abs(self.current_balance.subtlety_level - target_balance.subtlety_level)
        if subtlety_diff > 0.2:
            if self.current_balance.subtlety_level > target_balance.subtlety_level:
                adjustments["make_dynamics_clearer"] = True
                adjustments["increase_npc_assertions"] = True
            else:
                adjustments["hide_control_better"] = True
                adjustments["use_subtle_dynamics"] = True
        
        # Set cooldown if adjustments made
        if adjustments:
            self.adjustment_cooldown = 5
        
        return adjustments
    
    async def balance_event_generation(
        self,
        potential_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and balance potential events based on current state"""
        
        balanced_events = []
        
        for event in potential_events:
            score = self._score_event_balance(event)
            
            # Prefer events that improve balance
            if score > 0.5 or random.random() < 0.3:  # 30% chance for any event
                balanced_events.append(event)
        
        # Ensure minimum variety
        if len(balanced_events) < 3 and potential_events:
            # Add random events to ensure variety
            remaining = [e for e in potential_events if e not in balanced_events]
            if remaining:
                balanced_events.extend(random.sample(
                    remaining, 
                    min(3 - len(balanced_events), len(remaining))
                ))
        
        return balanced_events
    
    def _score_event_balance(self, event: Dict[str, Any]) -> float:
        """Score how well an event fits current balance needs"""
        score = 0.5  # Neutral baseline
        
        event_type = event.get("type", "")
        
        # Adjust based on current imbalances
        if self.current_balance.agency_level < 0.3:
            # Need more player agency
            if "choice" in event_type or "optional" in event:
                score += 0.2
        elif self.current_balance.agency_level > 0.7:
            # Need more NPC influence
            if "power" in event_type or "command" in event_type:
                score += 0.2
        
        if self.current_balance.routine_level < 0.4:
            # Need more routine
            if "routine" in event_type or "daily" in event_type:
                score += 0.15
        elif self.current_balance.routine_level > 0.8:
            # Need more variety
            if "special" in event_type or "random" in event_type:
                score += 0.15
        
        if self.current_balance.subtlety_level < 0.5:
            # Too overt, need subtlety
            if event.get("power_dynamic") in ["subtle_control", "protective_control"]:
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def adapt_npc_behavior(
        self,
        npc_id: int,
        base_behavior: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt NPC behavior based on balance needs"""
        
        adapted = base_behavior.copy()
        
        # Adjust dominance expression based on subtlety needs
        if self.current_balance.subtlety_level < 0.5:
            # Too overt, make more subtle
            if adapted.get("approach") == "direct_command":
                adapted["approach"] = "gentle_suggestion"
            adapted["intensity"] = adapted.get("intensity", 0.5) * 0.8
        elif self.current_balance.subtlety_level > 0.9:
            # Too subtle, make slightly clearer
            if adapted.get("approach") == "gentle_suggestion":
                adapted["approach"] = "clear_expectation"
            adapted["intensity"] = min(1.0, adapted.get("intensity", 0.5) * 1.2)
        
        # Adjust based on agency balance
        if self.current_balance.agency_level < 0.3:
            # Give player more choice
            adapted["allows_refusal"] = True
            adapted["alternative_options"] = True
        elif self.current_balance.agency_level > 0.7:
            # Assert more control
            adapted["expects_compliance"] = True
            adapted["consequences_for_refusal"] = True
        
        return adapted
    
    async def determine_scene_variety(
        self,
        recent_scenes: List[str]
    ) -> str:
        """Determine what type of scene to generate for variety"""
        
        if not recent_scenes:
            return "routine"
        
        # Count scene types
        scene_counts = {}
        for scene in recent_scenes[-10:]:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        
        # Find least used scene type
        all_types = ["routine", "social", "intimate", "work", "leisure", "special"]
        
        for scene_type in all_types:
            if scene_type not in scene_counts:
                return scene_type
        
        # Return least frequent
        return min(scene_counts.items(), key=lambda x: x[1])[0]
    
    def get_balance_report(self) -> Dict[str, Any]:
        """Get a report on current simulation balance"""
        
        recent_avg = SimulationBalance()
        if self.balance_history:
            recent = self.balance_history[-10:]
            recent_avg.agency_level = sum(b.agency_level for b in recent) / len(recent)
            recent_avg.routine_level = sum(b.routine_level for b in recent) / len(recent)
            recent_avg.subtlety_level = sum(b.subtlety_level for b in recent) / len(recent)
            recent_avg.emergence_level = sum(b.emergence_level for b in recent) / len(recent)
        
        return {
            "current": {
                "agency": self.current_balance.agency_level,
                "routine": self.current_balance.routine_level,
                "subtlety": self.current_balance.subtlety_level,
                "emergence": self.current_balance.emergence_level
            },
            "recent_average": {
                "agency": recent_avg.agency_level,
                "routine": recent_avg.routine_level,
                "subtlety": recent_avg.subtlety_level,
                "emergence": recent_avg.emergence_level
            },
            "recommendations": self._get_balance_recommendations()
        }
    
    def _get_balance_recommendations(self) -> List[str]:
        """Get recommendations for improving balance"""
        
        recommendations = []
        
        if self.current_balance.agency_level < 0.3:
            recommendations.append("Increase player choices and optional activities")
        elif self.current_balance.agency_level > 0.7:
            recommendations.append("Add more NPC-initiated interactions")
        
        if self.current_balance.routine_level < 0.4:
            recommendations.append("Establish more consistent daily routines")
        elif self.current_balance.routine_level > 0.8:
            recommendations.append("Introduce unexpected events and encounters")
        
        if self.current_balance.subtlety_level < 0.5:
            recommendations.append("Make power dynamics more subtle and natural")
        elif self.current_balance.subtlety_level > 0.9:
            recommendations.append("Allow some power dynamics to be more visible")
        
        if self.current_balance.emergence_level < 0.5:
            recommendations.append("Reduce scripted events, allow more emergence")
        
        return recommendations
