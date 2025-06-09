# logic/conflict_system/core.py
"""
Core conflict system functionality and data models
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from agents import RunContextWrapper
from db.connection import get_db_connection_context
from lore.core import canon

logger = logging.getLogger(__name__)

class ConflictPhase(Enum):
    BREWING = "brewing"
    ACTIVE = "active" 
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    CONCLUDED = "concluded"

class ConflictScale(Enum):
    PERSONAL = "personal"
    LOCAL = "local"
    REGIONAL = "regional"
    WORLD = "world"

class ResolutionApproach(Enum):
    DIPLOMATIC = "diplomatic"
    FORCEFUL = "forceful"
    MANIPULATIVE = "manipulative"
    INVESTIGATIVE = "investigative"
    SUBMISSIVE = "submissive"

@dataclass
class ConflictArchetype:
    """Defines a type of conflict and its characteristics"""
    name: str
    scale: ConflictScale
    min_stakeholders: int
    max_stakeholders: int
    base_duration: int
    facets: Dict[str, float]
    examples: List[str]

# Define conflict archetypes
CONFLICT_ARCHETYPES = {
    "personal_dispute": ConflictArchetype(
        name="personal_dispute",
        scale=ConflictScale.PERSONAL,
        min_stakeholders=2,
        max_stakeholders=4,
        base_duration=3,
        facets={"personal": 0.7, "political": 0.2, "crisis": 0.1},
        examples=["romantic rivalry", "friendship betrayal", "family feud"]
    ),
    "faction_rivalry": ConflictArchetype(
        name="faction_rivalry",
        scale=ConflictScale.LOCAL,
        min_stakeholders=4,
        max_stakeholders=8,
        base_duration=7,
        facets={"political": 0.6, "personal": 0.3, "crisis": 0.1},
        examples=["territory dispute", "resource competition", "ideological clash"]
    ),
    "succession_crisis": ConflictArchetype(
        name="succession_crisis",
        scale=ConflictScale.REGIONAL,
        min_stakeholders=5,
        max_stakeholders=10,
        base_duration=14,
        facets={"political": 0.7, "personal": 0.2, "mystery": 0.1},
        examples=["leadership vacuum", "contested inheritance", "coup attempt"]
    ),
    "economic_collapse": ConflictArchetype(
        name="economic_collapse",
        scale=ConflictScale.REGIONAL,
        min_stakeholders=6,
        max_stakeholders=12,
        base_duration=21,
        facets={"crisis": 0.7, "political": 0.3},
        examples=["trade war", "resource depletion", "currency crisis"]
    ),
    "religious_schism": ConflictArchetype(
        name="religious_schism",
        scale=ConflictScale.WORLD,
        min_stakeholders=8,
        max_stakeholders=15,
        base_duration=30,
        facets={"political": 0.4, "personal": 0.3, "mystery": 0.3},
        examples=["doctrine dispute", "prophet emergence", "sacred site conflict"]
    ),
    "apocalyptic_threat": ConflictArchetype(
        name="apocalyptic_threat",
        scale=ConflictScale.WORLD,
        min_stakeholders=10,
        max_stakeholders=20,
        base_duration=60,
        facets={"crisis": 0.8, "political": 0.2},
        examples=["plague outbreak", "environmental disaster", "ancient evil awakening"]
    )
}

class ConflictCore:
    """Core conflict system operations"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.ctx = RunContextWrapper({
            "user_id": user_id,
            "conversation_id": conversation_id
        })
    
    async def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """Get all active conflicts"""
        async with get_db_connection_context() as conn:
            conflicts = await conn.fetch("""
                SELECT c.*, 
                       COUNT(DISTINCT s.npc_id) as stakeholder_count,
                       COUNT(DISTINCT rp.path_id) as path_count
                FROM Conflicts c
                LEFT JOIN ConflictStakeholders s ON c.conflict_id = s.conflict_id
                LEFT JOIN ResolutionPaths rp ON c.conflict_id = rp.conflict_id
                WHERE c.user_id = $1 AND c.conversation_id = $2 
                    AND c.is_active = TRUE
                GROUP BY c.conflict_id
                ORDER BY c.conflict_id DESC
            """, self.user_id, self.conversation_id)
            
            return [dict(c) for c in conflicts]
    
    async def get_conflict_details(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific conflict"""
        async with get_db_connection_context() as conn:
            # Get base conflict
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, self.user_id, self.conversation_id)
            
            if not conflict:
                return None
                
            conflict_data = dict(conflict)
            
            # Get stakeholders
            stakeholders = await conn.fetch("""
                SELECT s.*, n.npc_name, n.dominance, n.cruelty
                FROM ConflictStakeholders s
                JOIN NPCStats n ON s.npc_id = n.npc_id
                WHERE s.conflict_id = $1
                ORDER BY s.involvement_level DESC
            """, conflict_id)
            conflict_data['stakeholders'] = [dict(s) for s in stakeholders]
            
            # Get resolution paths
            paths = await conn.fetch("""
                SELECT * FROM ResolutionPaths
                WHERE conflict_id = $1
                ORDER BY progress DESC
            """, conflict_id)
            conflict_data['resolution_paths'] = [dict(p) for p in paths]
            
            # Get player involvement
            involvement = await conn.fetchrow("""
                SELECT * FROM PlayerConflictInvolvement
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, self.user_id, self.conversation_id)
            conflict_data['player_involvement'] = dict(involvement) if involvement else None
            
            return conflict_data
    
    async def create_conflict(self, conflict_data: Dict[str, Any]) -> int:
        """Create a new conflict"""
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Get current day
                current_day = await self._get_current_day(conn)
                
                # Create conflict record
                conflict_id = await conn.fetchval("""
                    INSERT INTO Conflicts (
                        user_id, conversation_id, conflict_name, conflict_type,
                        description, progress, phase, start_day, estimated_duration,
                        success_rate, is_active
                    ) VALUES ($1, $2, $3, $4, $5, 0, $6, $7, $8, 0.5, TRUE)
                    RETURNING conflict_id
                """, 
                self.user_id, self.conversation_id,
                conflict_data['conflict_name'], 
                conflict_data.get('conflict_type', 'standard'),
                conflict_data['description'],
                ConflictPhase.BREWING.value,
                current_day,
                conflict_data.get('estimated_duration', 7)
                )
                
                # Log canonical event
                await canon.log_canonical_event(
                    self.ctx.context, conn,
                    f"Conflict emerged: {conflict_data['conflict_name']}",
                    tags=["conflict", "emergence", conflict_data.get('conflict_type', 'standard')],
                    significance=7
                )
                
                return conflict_id
    
    async def update_conflict_progress(self, conflict_id: int, 
                                     progress_delta: float) -> Dict[str, Any]:
        """Update conflict progress and handle phase transitions"""
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Get current state
                conflict = await conn.fetchrow("""
                    SELECT progress, phase FROM Conflicts
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, self.user_id, self.conversation_id)
                
                if not conflict:
                    return {"error": "Conflict not found"}
                
                old_progress = conflict['progress']
                old_phase = conflict['phase']
                new_progress = min(100, max(0, old_progress + progress_delta))
                
                # Determine phase based on progress
                new_phase = self._determine_phase(new_progress, old_phase)
                
                # Update conflict
                await conn.execute("""
                    UPDATE Conflicts
                    SET progress = $1, phase = $2, updated_at = NOW()
                    WHERE conflict_id = $3
                """, new_progress, new_phase, conflict_id)
                
                # Log phase change
                if old_phase != new_phase:
                    await canon.log_canonical_event(
                        self.ctx.context, conn,
                        f"Conflict phase changed from {old_phase} to {new_phase}",
                        tags=["conflict", "phase_change", new_phase],
                        significance=6
                    )
                
                return {
                    "old_progress": old_progress,
                    "new_progress": new_progress,
                    "old_phase": old_phase,
                    "new_phase": new_phase,
                    "phase_changed": old_phase != new_phase
                }
    
    def _determine_phase(self, progress: float, current_phase: str) -> str:
        """Determine conflict phase based on progress"""
        if progress >= 100:
            return ConflictPhase.CONCLUDED.value
        elif progress >= 90:
            return ConflictPhase.RESOLUTION.value
        elif progress >= 60:
            return ConflictPhase.CLIMAX.value
        elif progress >= 30:
            return ConflictPhase.ACTIVE.value
        else:
            return ConflictPhase.BREWING.value
    
    async def _get_current_day(self, conn) -> int:
        """Get current in-game day"""
        day = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentDay'
        """, self.user_id, self.conversation_id)
        return int(day) if day else 1
