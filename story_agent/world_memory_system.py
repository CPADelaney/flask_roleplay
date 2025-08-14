# story_agent/world_memory_system.py

"""
World Memory System for the slice-of-life simulation.
Tracks daily routines, relationships, and emergent patterns over time.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from db.connection import get_db_connection_context
from context.memory_manager import get_memory_manager
from logic.time_cycle import get_game_iso_string, get_current_time

logger = logging.getLogger(__name__)

@dataclass
class DailyRoutineMemory:
    """Memory of daily routines and patterns"""
    time_period: str  # morning, afternoon, evening, night
    common_activities: List[str] = field(default_factory=list)
    frequent_npcs: Dict[int, int] = field(default_factory=dict)  # NPC ID -> frequency
    power_dynamics_experienced: List[str] = field(default_factory=list)
    typical_mood: Optional[str] = None
    
    def add_activity(self, activity: str):
        if activity not in self.common_activities:
            self.common_activities.append(activity)
            if len(self.common_activities) > 10:
                self.common_activities.pop(0)
    
    def record_npc_interaction(self, npc_id: int):
        self.frequent_npcs[npc_id] = self.frequent_npcs.get(npc_id, 0) + 1

@dataclass
class RelationshipMemory:
    """Consolidated memory of a relationship"""
    npc_id: int
    npc_name: str
    first_interaction: datetime
    last_interaction: datetime
    total_interactions: int = 0
    
    # Power dynamic tracking
    submission_moments: List[str] = field(default_factory=list)
    resistance_moments: List[str] = field(default_factory=list)
    intimate_moments: List[str] = field(default_factory=list)
    
    # Pattern tracking
    common_activities: List[str] = field(default_factory=list)
    typical_dynamics: List[str] = field(default_factory=list)
    evolution_milestones: List[Dict[str, Any]] = field(default_factory=list)
    
    # FIXED: Single async method with proper parameters
    async def add_milestone(
        self,
        user_id: int,
        conversation_id: int,
        description: str,
        significance: float = 0.5
    ):
        """Add a milestone to this relationship's evolution"""
        timestamp = await get_game_iso_string(user_id, conversation_id)
        self.evolution_milestones.append({
            "description": description,
            "timestamp": timestamp,  # FIXED: Single timestamp key
            "game_time": await get_current_time(user_id, conversation_id),
            "significance": significance,
        })

class WorldMemorySystem:
    """Manages memories of the simulated world"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Memory structures
        self.daily_routines: Dict[str, DailyRoutineMemory] = {}
        self.relationships: Dict[int, RelationshipMemory] = {}
        self.world_patterns: List[str] = []
        self.significant_events: List[Dict[str, Any]] = []
        
        # Consolidation settings
        self.consolidation_interval = timedelta(hours=24)
        self.last_consolidation = datetime.now()
        
    async def initialize(self):
        """Load existing memories from database"""
        await self._load_daily_routines()
        await self._load_relationships()
        await self._load_patterns()
    
    async def _load_daily_routines(self):
        """Load daily routine memories"""
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT time_period, activities, npcs, dynamics, mood
                FROM DailyRoutineMemories
                WHERE user_id=$1 AND conversation_id=$2
            """, self.user_id, self.conversation_id)
        
        for row in rows:
            memory = DailyRoutineMemory(
                time_period=row['time_period'],
                common_activities=json.loads(row['activities']) if row['activities'] else [],
                frequent_npcs=json.loads(row['npcs']) if row['npcs'] else {},
                power_dynamics_experienced=json.loads(row['dynamics']) if row['dynamics'] else [],
                typical_mood=row['mood']
            )
            self.daily_routines[row['time_period']] = memory
    
    async def _load_relationships(self):
        """Load relationship memories"""
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, first_interaction, last_interaction,
                       total_interactions, submission_moments, resistance_moments,
                       intimate_moments, patterns, milestones
                FROM RelationshipMemories
                WHERE user_id=$1 AND conversation_id=$2
            """, self.user_id, self.conversation_id)
        
        for row in rows:
            memory = RelationshipMemory(
                npc_id=row['npc_id'],
                npc_name=row['npc_name'],
                first_interaction=row['first_interaction'],
                last_interaction=row['last_interaction'],
                total_interactions=row['total_interactions'],
                submission_moments=json.loads(row['submission_moments']) if row['submission_moments'] else [],
                resistance_moments=json.loads(row['resistance_moments']) if row['resistance_moments'] else [],
                intimate_moments=json.loads(row['intimate_moments']) if row['intimate_moments'] else [],
                evolution_milestones=json.loads(row['milestones']) if row['milestones'] else []
            )
            self.relationships[row['npc_id']] = memory
    
    async def _load_patterns(self):
        """Load world pattern memories"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT patterns, significant_events
                FROM WorldPatternMemories
                WHERE user_id=$1 AND conversation_id=$2
            """, self.user_id, self.conversation_id)
        
        if row:
            self.world_patterns = json.loads(row['patterns']) if row['patterns'] else []
            self.significant_events = json.loads(row['significant_events']) if row['significant_events'] else []
    
    async def record_daily_activity(
        self, 
        time_period: str,
        activity: str,
        involved_npcs: List[int],
        power_dynamic: Optional[str] = None,
        mood: Optional[str] = None
    ):
        """Record a daily activity"""
        
        if time_period not in self.daily_routines:
            self.daily_routines[time_period] = DailyRoutineMemory(time_period=time_period)
        
        routine = self.daily_routines[time_period]
        routine.add_activity(activity)
        
        for npc_id in involved_npcs:
            routine.record_npc_interaction(npc_id)
        
        if power_dynamic and power_dynamic not in routine.power_dynamics_experienced:
            routine.power_dynamics_experienced.append(power_dynamic)
        
        if mood:
            routine.typical_mood = mood
    
    async def record_relationship_event(
        self,
        npc_id: int,
        npc_name: str,
        event_type: str,
        description: str,
        is_milestone: bool = False
    ):
        """Record a relationship event"""
        
        if npc_id not in self.relationships:
            self.relationships[npc_id] = RelationshipMemory(
                npc_id=npc_id,
                npc_name=npc_name,
                first_interaction=datetime.now(),
                last_interaction=datetime.now()
            )
        
        relationship = self.relationships[npc_id]
        relationship.last_interaction = datetime.now()
        relationship.total_interactions += 1
        
        # Categorize event
        if "submit" in event_type.lower() or "accept" in event_type.lower():
            relationship.submission_moments.append(description[:100])
            if len(relationship.submission_moments) > 20:
                relationship.submission_moments.pop(0)
        elif "resist" in event_type.lower() or "refuse" in event_type.lower():
            relationship.resistance_moments.append(description[:100])
            if len(relationship.resistance_moments) > 20:
                relationship.resistance_moments.pop(0)
        elif "intimate" in event_type.lower() or "close" in event_type.lower():
            relationship.intimate_moments.append(description[:100])
            if len(relationship.intimate_moments) > 20:
                relationship.intimate_moments.pop(0)
        
        # FIXED: Proper await call with all required parameters
        if is_milestone:
            await relationship.add_milestone(
                self.user_id, self.conversation_id, description
            )
    
    async def detect_patterns(self) -> List[str]:
        """Detect patterns in world memories"""
        patterns = []
        
        # Check daily routine patterns
        morning_routine = self.daily_routines.get("morning")
        if morning_routine and len(morning_routine.common_activities) > 5:
            patterns.append(f"Established morning routine: {', '.join(morning_routine.common_activities[:3])}")
        
        # Check relationship patterns
        for npc_id, relationship in self.relationships.items():
            submission_ratio = len(relationship.submission_moments) / max(1, 
                len(relationship.submission_moments) + len(relationship.resistance_moments))
            
            if submission_ratio > 0.7 and relationship.total_interactions > 10:
                patterns.append(f"Submissive pattern with {relationship.npc_name}")
            elif submission_ratio < 0.3 and relationship.total_interactions > 10:
                patterns.append(f"Resistant pattern with {relationship.npc_name}")
        
        # Check power dynamic patterns
        all_dynamics = []
        for routine in self.daily_routines.values():
            all_dynamics.extend(routine.power_dynamics_experienced)
        
        if all_dynamics:
            most_common = max(set(all_dynamics), key=all_dynamics.count)
            if all_dynamics.count(most_common) > 5:
                patterns.append(f"Frequently experiencing {most_common}")
        
        self.world_patterns = patterns
        return patterns
    
    async def consolidate_memories(self):
        """Consolidate short-term memories into long-term patterns"""
        
        # Check if it's time to consolidate
        if datetime.now() - self.last_consolidation < self.consolidation_interval:
            return
        
        patterns = await self.detect_patterns()
        
        # Create consolidated memory entry
        memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        
        # Consolidate daily routines
        for time_period, routine in self.daily_routines.items():
            if routine.common_activities:
                summary = f"{time_period} routine: {', '.join(routine.common_activities[:3])}"
                await memory_manager.add_memory(
                    content=summary,
                    memory_type="routine_consolidation",
                    importance=0.4,
                    tags=["daily_routine", time_period]
                )
        
        # Consolidate relationships
        for npc_id, relationship in self.relationships.items():
            if relationship.total_interactions > 5:
                summary = self._summarize_relationship(relationship)
                await memory_manager.add_memory(
                    content=summary,
                    memory_type="relationship_consolidation",
                    importance=0.6,
                    tags=["relationship", f"npc_{npc_id}"]
                )
        
        # Save patterns
        if patterns:
            await memory_manager.add_memory(
                content=f"Observed patterns: {'; '.join(patterns)}",
                memory_type="pattern_recognition",
                importance=0.7,
                tags=["patterns", "world_state"]
            )
        
        self.last_consolidation = datetime.now()
        await self._save_state()
    
    def _summarize_relationship(self, relationship: RelationshipMemory) -> str:
        """Create a summary of a relationship"""
        duration = (relationship.last_interaction - relationship.first_interaction).days
        
        # Determine relationship character
        submission_count = len(relationship.submission_moments)
        resistance_count = len(relationship.resistance_moments)
        intimate_count = len(relationship.intimate_moments)
        
        if submission_count > resistance_count * 2:
            character = "submissive"
        elif resistance_count > submission_count * 2:
            character = "resistant"
        elif intimate_count > (submission_count + resistance_count) / 2:
            character = "intimate"
        else:
            character = "complex"
        
        summary = (f"Relationship with {relationship.npc_name}: "
                  f"{duration} days, {relationship.total_interactions} interactions, "
                  f"characterized as {character}")
        
        if relationship.evolution_milestones:
            latest_milestone = relationship.evolution_milestones[-1]
            summary += f". Latest: {latest_milestone['description']}"
        
        return summary
    
    async def get_daily_summary(self) -> str:
        """Get a summary of the day's activities"""
        summaries = []
        
        for time_period in ["morning", "afternoon", "evening", "night"]:
            if time_period in self.daily_routines:
                routine = self.daily_routines[time_period]
                if routine.common_activities:
                    summaries.append(f"{time_period.title()}: {routine.common_activities[-1]}")
        
        if summaries:
            return "Today's activities - " + "; ".join(summaries)
        return "A quiet day with no notable activities"
    
    async def get_relationship_summary(self, npc_id: int) -> Optional[str]:
        """Get a summary of a specific relationship"""
        if npc_id not in self.relationships:
            return None
        
        return self._summarize_relationship(self.relationships[npc_id])
    
    async def _save_state(self):
        """Save memory state to database"""
        async with get_db_connection_context() as conn:
            # Save daily routines
            for time_period, routine in self.daily_routines.items():
                await conn.execute("""
                    INSERT INTO DailyRoutineMemories 
                    (user_id, conversation_id, time_period, activities, npcs, dynamics, mood)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (user_id, conversation_id, time_period)
                    DO UPDATE SET 
                        activities = $4, npcs = $5, dynamics = $6, mood = $7
                """, self.user_id, self.conversation_id, time_period,
                    json.dumps(routine.common_activities),
                    json.dumps(routine.frequent_npcs),
                    json.dumps(routine.power_dynamics_experienced),
                    routine.typical_mood)
            
            # Save relationships
            for npc_id, relationship in self.relationships.items():
                await conn.execute("""
                    INSERT INTO RelationshipMemories
                    (user_id, conversation_id, npc_id, npc_name, first_interaction,
                     last_interaction, total_interactions, submission_moments,
                     resistance_moments, intimate_moments, milestones)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (user_id, conversation_id, npc_id)
                    DO UPDATE SET
                        last_interaction = $6, total_interactions = $7,
                        submission_moments = $8, resistance_moments = $9,
                        intimate_moments = $10, milestones = $11
                """, self.user_id, self.conversation_id, npc_id, relationship.npc_name,
                    relationship.first_interaction, relationship.last_interaction,
                    relationship.total_interactions,
                    json.dumps(relationship.submission_moments[-20:]),
                    json.dumps(relationship.resistance_moments[-20:]),
                    json.dumps(relationship.intimate_moments[-20:]),
                    json.dumps(relationship.evolution_milestones[-10:]))
            
            # Save patterns
            await conn.execute("""
                INSERT INTO WorldPatternMemories
                (user_id, conversation_id, patterns, significant_events, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, conversation_id)
                DO UPDATE SET
                    patterns = $3, significant_events = $4, updated_at = $5
            """, self.user_id, self.conversation_id,
                json.dumps(self.world_patterns),
                json.dumps(self.significant_events[-50:]),
                datetime.now())
