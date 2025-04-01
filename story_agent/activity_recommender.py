"""
Activity Recommendation Agent - Analyzes current scene context and NPCs to suggest
appropriate activities from the available options.
"""

import random
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from utils.npc_utils import get_npc_personality_traits
from utils.story_context import get_current_scenario_context
from utils.memory_utils import get_relevant_memories
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

@dataclass
class ActivityContext:
    """Context for activity recommendation"""
    present_npcs: List[Dict]  # List of NPC data
    location: str
    time_of_day: str
    current_mood: str
    recent_activities: List[str]
    scenario_type: str
    relationship_levels: Dict[str, int]  # NPC ID to relationship level

@dataclass
class ActivityRecommendation:
    """Represents a recommended activity"""
    activity_name: str
    confidence_score: float  # 0-1
    reasoning: str
    participating_npcs: List[str]
    estimated_duration: str
    prerequisites: List[str]
    expected_outcomes: List[str]

class ActivityRecommender:
    """Recommends contextually appropriate activities"""
    
    def __init__(self):
        # Activity categories and their base weights
        self.activity_categories = {
            "social": 1.0,
            "training": 1.0,
            "service": 1.0,
            "relaxation": 1.0,
            "challenge": 1.0
        }
        
        # Factors that influence activity selection
        self.influence_factors = {
            "npc_mood": {
                "happy": {"social": 1.3, "relaxation": 1.2},
                "strict": {"training": 1.3, "challenge": 1.2},
                "stressed": {"relaxation": 1.4, "social": 0.8},
                "focused": {"training": 1.2, "challenge": 1.3}
            },
            "time_of_day": {
                "morning": {"training": 1.3, "challenge": 1.2},
                "afternoon": {"social": 1.2, "service": 1.2},
                "evening": {"relaxation": 1.3, "social": 1.2},
                "night": {"relaxation": 1.4, "training": 0.8}
            },
            "relationship_level": {
                "low": {"service": 1.3, "training": 1.2},
                "medium": {"social": 1.2, "challenge": 1.2},
                "high": {"relaxation": 1.2, "social": 1.3}
            }
        }
    
    async def _get_activity_context(self, scenario_id: str, npc_ids: List[str]) -> ActivityContext:
        """Gather context for activity recommendation"""
        scenario = await get_current_scenario_context(scenario_id)
        npcs = [await get_npc_personality_traits(npc_id) for npc_id in npc_ids]
        
        # This would be more detailed in actual implementation
        return ActivityContext(
            present_npcs=npcs,
            location=scenario["location"],
            time_of_day=scenario["time_of_day"],
            current_mood=scenario["mood"],
            recent_activities=scenario["recent_activities"],
            scenario_type=scenario["type"],
            relationship_levels={npc["id"]: npc["relationship_level"] for npc in npcs}
        )
    
    def _calculate_activity_weights(self, context: ActivityContext) -> Dict[str, float]:
        """Calculate weights for each activity category based on context"""
        weights = self.activity_categories.copy()
        
        # Apply NPC mood influences
        for npc in context.present_npcs:
            mood = npc.get("current_mood", "neutral")
            if mood in self.influence_factors["npc_mood"]:
                modifiers = self.influence_factors["npc_mood"][mood]
                for category, modifier in modifiers.items():
                    weights[category] *= modifier
        
        # Apply time of day influences
        if context.time_of_day in self.influence_factors["time_of_day"]:
            modifiers = self.influence_factors["time_of_day"][context.time_of_day]
            for category, modifier in modifiers.items():
                weights[category] *= modifier
        
        # Apply relationship level influences
        avg_relationship = sum(context.relationship_levels.values()) / len(context.relationship_levels)
        rel_level = "low" if avg_relationship < 30 else "medium" if avg_relationship < 70 else "high"
        modifiers = self.influence_factors["relationship_level"][rel_level]
        for category, modifier in modifiers.items():
            weights[category] *= modifier
        
        return weights
    
    def _filter_activities(self, activities: List[Dict], context: ActivityContext) -> List[Dict]:
        """Filter activities based on context"""
        filtered = []
        
        for activity in activities:
            # Check prerequisites
            if not self._check_prerequisites(activity, context):
                continue
                
            # Check if recently done
            if activity["name"] in context.recent_activities[-5:]:
                continue
                
            # Check location compatibility
            if not self._is_location_compatible(activity, context.location):
                continue
                
            filtered.append(activity)
        
        return filtered
    
    def _check_prerequisites(self, activity: Dict, context: ActivityContext) -> bool:
        """Check if activity prerequisites are met"""
        # This would be more detailed in actual implementation
        return True
    
    def _is_location_compatible(self, activity: Dict, location: str) -> bool:
        """Check if activity can be done in current location"""
        # This would be more detailed in actual implementation
        return True
    
    def _score_activity(self, activity: Dict, category_weights: Dict[str, float], context: ActivityContext) -> float:
        """Score an activity based on context and weights"""
        base_score = category_weights.get(activity["category"], 1.0)
        
        # Adjust for NPC compatibility
        npc_compatibility = self._calculate_npc_compatibility(activity, context.present_npcs)
        
        # Adjust for variety (lower score if similar activities were recent)
        variety_factor = self._calculate_variety_factor(activity, context.recent_activities)
        
        # Adjust for time appropriateness
        time_factor = self._calculate_time_factor(activity, context.time_of_day)
        
        return base_score * npc_compatibility * variety_factor * time_factor
    
    def _calculate_npc_compatibility(self, activity: Dict, npcs: List[Dict]) -> float:
        """Calculate how well the activity matches present NPCs"""
        compatibility_scores = []
        
        for npc in npcs:
            score = 1.0
            
            # Adjust based on NPC traits
            for trait in npc.get("traits", []):
                if trait.lower() in activity.get("preferred_traits", []):
                    score *= 1.2
                elif trait.lower() in activity.get("avoided_traits", []):
                    score *= 0.8
            
            compatibility_scores.append(score)
        
        return sum(compatibility_scores) / len(compatibility_scores)
    
    def _calculate_variety_factor(self, activity: Dict, recent_activities: List[str]) -> float:
        """Calculate variety factor to encourage diverse activities"""
        if activity["name"] in recent_activities:
            return 0.7
        elif activity["category"] in [act.split(':')[0] for act in recent_activities]:
            return 0.9
        return 1.0
    
    def _calculate_time_factor(self, activity: Dict, time_of_day: str) -> float:
        """Calculate how appropriate the activity is for the time of day"""
        preferred_times = activity.get("preferred_times", [])
        if not preferred_times or time_of_day in preferred_times:
            return 1.0
        return 0.8
    
    def _generate_reasoning(self, activity: Dict, score: float, context: ActivityContext) -> str:
        """Generate explanation for why this activity was recommended"""
        reasons = []
        
        # Add category-based reason
        reasons.append(f"Matches current scenario type: {context.scenario_type}")
        
        # Add NPC-based reason
        compatible_npcs = [
            npc["name"] for npc in context.present_npcs
            if any(trait.lower() in activity.get("preferred_traits", [])
                  for trait in npc.get("traits", []))
        ]
        if compatible_npcs:
            reasons.append(f"Well-suited for {', '.join(compatible_npcs)}")
        
        # Add timing-based reason
        if context.time_of_day in activity.get("preferred_times", []):
            reasons.append(f"Ideal for {context.time_of_day} activities")
        
        # Add variety-based reason
        if activity["name"] not in context.recent_activities:
            reasons.append("Provides variety from recent activities")
        
        return " | ".join(reasons)
    
    async def recommend_activities(
        self,
        scenario_id: str,
        npc_ids: List[str],
        available_activities: List[Dict],
        num_recommendations: int = 2
    ) -> List[ActivityRecommendation]:
        """Generate activity recommendations"""
        # Get context
        context = await self._get_activity_context(scenario_id, npc_ids)
        
        # Calculate category weights
        category_weights = self._calculate_activity_weights(context)
        
        # Filter activities
        filtered_activities = self._filter_activities(available_activities, context)
        
        # Score activities
        scored_activities = []
        for activity in filtered_activities:
            score = self._score_activity(activity, category_weights, context)
            scored_activities.append((activity, score))
        
        # Sort by score
        scored_activities.sort(key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        recommendations = []
        for activity, score in scored_activities[:num_recommendations]:
            recommendation = ActivityRecommendation(
                activity_name=activity["name"],
                confidence_score=min(1.0, score),
                reasoning=self._generate_reasoning(activity, score, context),
                participating_npcs=[npc["name"] for npc in context.present_npcs],
                estimated_duration=activity.get("duration", "30 minutes"),
                prerequisites=activity.get("prerequisites", []),
                expected_outcomes=activity.get("outcomes", [])
            )
            recommendations.append(recommendation)
        
        # Add "none" as the third option if requested
        if len(recommendations) < 3:
            none_recommendation = ActivityRecommendation(
                activity_name="None",
                confidence_score=0.5,
                reasoning="Alternative option if other activities don't appeal",
                participating_npcs=[],
                estimated_duration="N/A",
                prerequisites=[],
                expected_outcomes=["Maintain current activity or free choice"]
            )
            recommendations.append(none_recommendation)
        
        return recommendations

# Example usage:
"""
recommender = ActivityRecommender()
activities = [
    {
        "name": "Training Session",
        "category": "training",
        "preferred_traits": ["disciplined", "focused"],
        "avoided_traits": ["lazy"],
        "preferred_times": ["morning", "afternoon"],
        "prerequisites": ["training equipment"],
        "outcomes": ["skill improvement", "increased discipline"]
    },
    # More activities...
]

recommendations = recommender.recommend_activities(
    "scenario123",
    ["npc456", "npc789"],
    activities
)

for rec in recommendations:
    print(f"\nActivity: {rec.activity_name}")
    print(f"Confidence: {rec.confidence_score:.2f}")
    print(f"Reasoning: {rec.reasoning}")
    print(f"Duration: {rec.estimated_duration}")
    print(f"Participants: {', '.join(rec.participating_npcs)}")
""" 
