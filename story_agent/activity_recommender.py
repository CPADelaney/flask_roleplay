# story_agent/activity_recommender.py

"""
Activity Recommendation Agent - Analyzes current scene context and NPCs to suggest
appropriate activities from the available options.
"""

import random
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Database connection
from db.connection import get_db_connection_context

# Context retrieval
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager
from logic.aggregator_sdk import get_aggregated_roleplay_context

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
    
    async def _get_npc_personality_traits(self, user_id: int, conversation_id: int, npc_id: int) -> Dict:
        """Get NPC personality traits directly from database"""
        async with get_db_connection_context() as conn:
            # Query the NPCStats table to get personality traits
            query = """
                SELECT npc_id, npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                       personality_traits, hobbies, likes, dislikes, current_location
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """
            row = await conn.fetchrow(query, user_id, conversation_id, npc_id)
            
            if not row:
                return {
                    "id": npc_id,
                    "name": "Unknown",
                    "traits": [],
                    "relationship_level": 0,
                    "current_mood": "neutral",
                    "role": "unknown"
                }
            
            # Parse JSON fields
            try:
                personality_traits = json.loads(row["personality_traits"]) if isinstance(row["personality_traits"], str) else row["personality_traits"] or []
            except (json.JSONDecodeError, TypeError):
                personality_traits = []
                
            try:
                hobbies = json.loads(row["hobbies"]) if isinstance(row["hobbies"], str) else row["hobbies"] or []
            except (json.JSONDecodeError, TypeError):
                hobbies = []
                
            try:
                likes = json.loads(row["likes"]) if isinstance(row["likes"], str) else row["likes"] or []
            except (json.JSONDecodeError, TypeError):
                likes = []
                
            try:
                dislikes = json.loads(row["dislikes"]) if isinstance(row["dislikes"], str) else row["dislikes"] or []
            except (json.JSONDecodeError, TypeError):
                dislikes = []
            
            # Estimate mood based on stats
            mood = "neutral"
            if row["intensity"] > 75:
                mood = "focused"
            elif row["dominance"] > 75:
                mood = "strict"
            elif row["cruelty"] > 75:
                mood = "stressed"
            elif row["trust"] > 75 and row["respect"] > 75:
                mood = "happy"
            
            # Determine role based on stats
            role = "neutral"
            if row["dominance"] > 70:
                role = "dominant"
            elif row["closeness"] > 70:
                role = "close"
            
            # Get relationship level (average of closeness, trust, respect)
            relationship_level = int((row["closeness"] + row["trust"] + row["respect"]) / 3)
            
            return {
                "id": row["npc_id"],
                "name": row["npc_name"],
                "traits": personality_traits,
                "hobbies": hobbies,
                "likes": likes,
                "dislikes": dislikes,
                "stats": {
                    "dominance": row["dominance"],
                    "cruelty": row["cruelty"],
                    "closeness": row["closeness"],
                    "trust": row["trust"],
                    "respect": row["respect"],
                    "intensity": row["intensity"]
                },
                "relationship_level": relationship_level,
                "current_mood": mood,
                "role": role,
                "current_location": row["current_location"]
            }
    
    async def _get_current_scenario_context(self, user_id: int, conversation_id: int) -> Dict:
        """Get current scenario context using aggregator"""
        try:
            # Try using the context service if available
            context_service = await get_context_service(user_id, conversation_id)
            context_data = await context_service.get_context()
            
            scenario_type = "default"
            if "current_conflict" in context_data and context_data["current_conflict"]:
                scenario_type = "conflict"
            elif "current_event" in context_data and context_data["current_event"]:
                scenario_type = "event"
            
            # Extract recent activities from messages
            recent_activities = []
            async with get_db_connection_context() as conn:
                # Get last 5 user messages
                rows = await conn.fetch("""
                    SELECT content FROM messages
                    WHERE conversation_id = $1 AND sender = 'user'
                    ORDER BY created_at DESC
                    LIMIT 5
                """, conversation_id)
                
                for row in rows:
                    recent_activities.append(row["content"])
            
            return {
                "type": scenario_type,
                "location": context_data.get("current_location", "Unknown"),
                "time_of_day": context_data.get("time_of_day", "Morning"),
                "mood": "neutral",  # Default mood
                "player_status": context_data.get("player_stats", {}),
                "recent_activities": recent_activities
            }
            
        except Exception as e:
            logger.warning(f"Error getting context from context service: {e}, falling back to aggregator")
            
            # Fallback to using the aggregator
            aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id)
            
            # Determine scenario type
            scenario_type = "default"
            current_roleplay = aggregator_data.get("current_roleplay", {})
            if "CurrentConflict" in current_roleplay and current_roleplay["CurrentConflict"]:
                scenario_type = "conflict"
            elif "CurrentEvent" in current_roleplay and current_roleplay["CurrentEvent"]:
                scenario_type = "event"
            
            # Get recent activities
            recent_activities = []
            async with get_db_connection_context() as conn:
                # Get last 5 user messages
                rows = await conn.fetch("""
                    SELECT content FROM messages
                    WHERE conversation_id = $1 AND sender = 'user'
                    ORDER BY created_at DESC
                    LIMIT 5
                """, conversation_id)
                
                for row in rows:
                    recent_activities.append(row["content"])
            
            return {
                "type": scenario_type,
                "location": aggregator_data.get("current_location", "Unknown"),
                "time_of_day": aggregator_data.get("time_of_day", "Morning"),
                "mood": "neutral",  # Default mood
                "player_status": aggregator_data.get("player_stats", {}),
                "recent_activities": recent_activities
            }
    
    async def _get_relevant_memories(self, user_id: int, conversation_id: int, npc_id: int, limit: int = 5) -> List[Dict]:
        """Get relevant memories using the memory manager"""
        try:
            # Try to use memory manager from context system
            memory_manager = await get_memory_manager(user_id, conversation_id)
            
            # Get memories for this NPC
            async with get_db_connection_context() as conn:
                # Get NPC name
                row = await conn.fetchrow("""
                    SELECT npc_name FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                """, user_id, conversation_id, npc_id)
                
                npc_name = row["npc_name"] if row else f"NPC-{npc_id}"
            
            # Search memories with vector search if available
            memories = await memory_manager.search_memories(
                query_text=f"activities with {npc_name}",
                limit=limit,
                tags=[npc_name.lower().replace(" ", "_")],
                use_vector=True
            )
            
            # Format memories
            memory_dicts = []
            for memory in memories:
                if hasattr(memory, 'to_dict'):
                    memory_dicts.append(memory.to_dict())
                else:
                    # If it's already a dict or another format
                    memory_dicts.append(memory)
            
            return memory_dicts
            
        except Exception as e:
            logger.warning(f"Error getting memories from memory manager: {e}, falling back to database")
            
            # Fallback to direct database query
            memories = []
            async with get_db_connection_context() as conn:
                # Try unified memories table first
                try:
                    rows = await conn.fetch("""
                        SELECT memory_text, memory_type, timestamp
                        FROM unified_memories
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND entity_type = 'npc' AND entity_id = $3
                        ORDER BY timestamp DESC
                        LIMIT $4
                    """, user_id, conversation_id, npc_id, limit)
                    
                    for row in rows:
                        memories.append({
                            "content": row["memory_text"],
                            "type": row["memory_type"],
                            "timestamp": row["timestamp"].isoformat()
                        })
                        
                except Exception as e2:
                    logger.warning(f"Error querying unified_memories: {e2}, trying NPCMemories")
                    
                    # Try legacy NPCMemories table
                    try:
                        rows = await conn.fetch("""
                            SELECT memory_text, memory_type, timestamp
                            FROM NPCMemories
                            WHERE npc_id = $1
                            ORDER BY timestamp DESC
                            LIMIT $2
                        """, npc_id, limit)
                        
                        for row in rows:
                            memories.append({
                                "content": row["memory_text"],
                                "type": row["memory_type"],
                                "timestamp": row["timestamp"].isoformat()
                            })
                    except Exception as e3:
                        logger.error(f"Error querying NPCMemories: {e3}")
            
            return memories
    
    async def _get_activity_context(self, user_id: int, conversation_id: int, scenario_id: str, npc_ids: List[int]) -> ActivityContext:
        """Gather context for activity recommendation"""
        # Get scenario context
        scenario = await self._get_current_scenario_context(user_id, conversation_id)
        
        # Get NPC data
        npcs = []
        for npc_id in npc_ids:
            npc_data = await self._get_npc_personality_traits(user_id, conversation_id, npc_id)
            npcs.append(npc_data)
        
        # Build relationship levels dict
        relationship_levels = {
            str(npc["id"]): npc["relationship_level"] for npc in npcs
        }
        
        return ActivityContext(
            present_npcs=npcs,
            location=scenario["location"],
            time_of_day=scenario["time_of_day"],
            current_mood=scenario["mood"],
            recent_activities=scenario["recent_activities"],
            scenario_type=scenario["type"],
            relationship_levels=relationship_levels
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
        time_of_day = context.time_of_day.lower()
        if time_of_day in self.influence_factors["time_of_day"]:
            modifiers = self.influence_factors["time_of_day"][time_of_day]
            for category, modifier in modifiers.items():
                weights[category] *= modifier
        
        # Apply relationship level influences
        avg_relationship = sum(context.relationship_levels.values()) / len(context.relationship_levels) if context.relationship_levels else 50
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
        
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 1.0
    
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
        if not preferred_times or time_of_day.lower() in [t.lower() for t in preferred_times]:
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
        if context.time_of_day.lower() in [t.lower() for t in activity.get("preferred_times", [])]:
            reasons.append(f"Ideal for {context.time_of_day} activities")
        
        # Add variety-based reason
        if activity["name"] not in context.recent_activities:
            reasons.append("Provides variety from recent activities")
        
        return " | ".join(reasons)
    
    async def _get_available_activities(self, user_id: int, conversation_id: int) -> List[Dict]:
        """Get available activities from the Activities table"""
        async with get_db_connection_context() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT name, purpose, stat_integration, intensity_tiers, setting_variants, fantasy_level
                    FROM Activities
                """)
                
                activities = []
                for row in rows:
                    # Parse JSON fields
                    try:
                        purpose = json.loads(row["purpose"]) if isinstance(row["purpose"], str) else row["purpose"] or {}
                    except (json.JSONDecodeError, TypeError):
                        purpose = {}
                        
                    try:
                        stat_integration = json.loads(row["stat_integration"]) if isinstance(row["stat_integration"], str) else row["stat_integration"] or {}
                    except (json.JSONDecodeError, TypeError):
                        stat_integration = {}
                        
                    try:
                        intensity_tiers = json.loads(row["intensity_tiers"]) if isinstance(row["intensity_tiers"], str) else row["intensity_tiers"] or {}
                    except (json.JSONDecodeError, TypeError):
                        intensity_tiers = {}
                        
                    try:
                        setting_variants = json.loads(row["setting_variants"]) if isinstance(row["setting_variants"], str) else row["setting_variants"] or {}
                    except (json.JSONDecodeError, TypeError):
                        setting_variants = {}
                    
                    # Determine category based on purpose or name
                    category = "social"
                    if "training" in purpose or "skill" in purpose:
                        category = "training"
                    elif "service" in purpose or "help" in purpose:
                        category = "service"
                    elif "relaxation" in purpose or "leisure" in purpose:
                        category = "relaxation"
                    elif "challenge" in purpose or "test" in purpose:
                        category = "challenge"
                    
                    # Extract preferred traits from purpose data
                    preferred_traits = []
                    avoided_traits = []
                    
                    if purpose.get("compatibility"):
                        preferred_traits = purpose["compatibility"].get("preferred", [])
                        avoided_traits = purpose["compatibility"].get("avoided", [])
                    
                    # Extract preferred times
                    preferred_times = []
                    
                    if purpose.get("timing"):
                        preferred_times = purpose["timing"].get("preferred_times", [])
                    
                    activities.append({
                        "name": row["name"],
                        "category": category,
                        "purpose": purpose,
                        "stat_integration": stat_integration,
                        "intensity_tiers": intensity_tiers,
                        "setting_variants": setting_variants,
                        "fantasy_level": row["fantasy_level"],
                        "preferred_traits": preferred_traits,
                        "avoided_traits": avoided_traits,
                        "preferred_times": preferred_times,
                        "prerequisites": purpose.get("prerequisites", []),
                        "outcomes": stat_integration.get("outcomes", [])
                    })
                
                return activities
                
            except Exception as e:
                logger.error(f"Error fetching activities: {e}")
                
                # Return a small set of default activities
                return [
                    {
                        "name": "Training Session",
                        "category": "training",
                        "preferred_traits": ["disciplined", "focused"],
                        "avoided_traits": ["lazy"],
                        "preferred_times": ["morning", "afternoon"],
                        "prerequisites": ["training equipment"],
                        "outcomes": ["skill improvement", "increased discipline"]
                    },
                    {
                        "name": "Relaxation Time",
                        "category": "relaxation",
                        "preferred_traits": ["calm", "patient"],
                        "avoided_traits": ["anxious"],
                        "preferred_times": ["evening", "night"],
                        "prerequisites": [],
                        "outcomes": ["reduced stress", "improved mood"]
                    },
                    {
                        "name": "Social Gathering",
                        "category": "social",
                        "preferred_traits": ["outgoing", "friendly"],
                        "avoided_traits": ["antisocial"],
                        "preferred_times": ["afternoon", "evening"],
                        "prerequisites": [],
                        "outcomes": ["improved relationships", "social information"]
                    }
                ]
    
    async def recommend_activities(
        self,
        user_id: int,
        conversation_id: int,
        scenario_id: str,
        npc_ids: List[int],
        num_recommendations: int = 2
    ) -> List[ActivityRecommendation]:
        """Generate activity recommendations"""
        if not npc_ids:
            # Get some available NPCs to recommend activities with
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
                    LIMIT 3
                """, user_id, conversation_id)
                
                npc_ids = [row["npc_id"] for row in rows]
        
        # Get context
        context = await self._get_activity_context(user_id, conversation_id, scenario_id, npc_ids)
        
        # Get available activities
        available_activities = await self._get_available_activities(user_id, conversation_id)
        
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
activities = await recommender.recommend_activities(
    user_id=123,
    conversation_id=456,
    scenario_id="scenario123",
    npc_ids=[789, 790]
)

for rec in activities:
    print(f"\nActivity: {rec.activity_name}")
    print(f"Confidence: {rec.confidence_score:.2f}")
    print(f"Reasoning: {rec.reasoning}")
    print(f"Duration: {rec.estimated_duration}")
    print(f"Participants: {', '.join(rec.participating_npcs)}")
"""
