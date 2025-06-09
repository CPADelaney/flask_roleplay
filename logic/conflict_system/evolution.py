# logic/conflict_system/evolution.py
"""
Conflict evolution and dynamic progression
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

from agents import Agent, Runner, ModelSettings
from db.connection import get_db_connection_context

from .core import ConflictCore, ConflictPhase
from .stakeholders import StakeholderManager

logger = logging.getLogger(__name__)

# Evolution Strategy Agent
evolution_agent = Agent(
    name="Conflict Evolution Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.7),
    instructions="""
    You manage how conflicts evolve based on events and actions.
    
    Consider:
    - Current conflict phase and progress
    - Recent player actions
    - Stakeholder autonomous actions
    - External events
    - Natural escalation patterns
    
    Determine:
    - Progress changes
    - Phase transitions
    - New complications
    - Stakeholder reactions
    - Emerging opportunities
    
    Ensure evolution feels natural and responsive.
    Output specific updates as JSON.
    """
)

class ConflictEvolution:
    """Manages dynamic conflict progression"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.core = ConflictCore(user_id, conversation_id)
        self.stakeholders = StakeholderManager(user_id, conversation_id)
    
    async def evolve_conflict(self, conflict_id: int,
                            event_type: str,
                            event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve a conflict based on an event"""
        # Get current state
        conflict = await self.core.get_conflict_details(conflict_id)
        if not conflict:
            return {"error": "Conflict not found"}
        
        # Analyze event impact
        impact = await self._analyze_event_impact(
            conflict, event_type, event_data
        )
        
        # Apply evolution
        evolution_result = await self._apply_evolution(
            conflict_id, conflict, impact
        )
        
        # Process stakeholder reactions
        reactions = await self.stakeholders.process_stakeholder_reactions(
            conflict_id, evolution_result
        )
        
        return {
            "conflict_id": conflict_id,
            "event_type": event_type,
            "evolution": evolution_result,
            "stakeholder_reactions": reactions
        }
    
    async def process_natural_evolution(self, conflict_id: int) -> Dict[str, Any]:
        """Process natural conflict evolution over time"""
        conflict = await self.core.get_conflict_details(conflict_id)
        if not conflict:
            return {"error": "Conflict not found"}
        
        # Check if evolution needed
        if not await self._needs_natural_evolution(conflict):
            return {"evolved": False}
        
        # Determine natural progression
        event_data = {
            "days_active": await self._get_conflict_age(conflict_id),
            "stagnation": conflict['progress'] < 30 and conflict['phase'] == 'brewing'
        }
        
        return await self.evolve_conflict(
            conflict_id, "natural_progression", event_data
        )
    
    async def _analyze_event_impact(self, conflict: Dict[str, Any],
                                  event_type: str,
                                  event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how an event impacts the conflict"""
        prompt = f"""
        Analyze impact of this event on the conflict:
        
        Conflict: {conflict['conflict_name']} 
        Phase: {conflict['phase']} ({conflict['progress']}% progress)
        
        Event Type: {event_type}
        Event Data: {json.dumps(event_data, indent=2)}
        
        Stakeholders:
        {json.dumps([{
            'name': s['npc_name'],
            'involvement': s['involvement_level'],
            'motivation': s['public_motivation']
        } for s in conflict['stakeholders'][:5]], indent=2)}
        
        Determine:
        1. Progress change (-20 to +30)
        2. Stakeholder position changes
        3. New complications (if any)
        4. Resolution path impacts
        
        Output as JSON.
        """
        
        result = await Runner.run(
            evolution_agent,
            prompt,
            self.core.ctx
        )
        
        return json.loads(result.final_output)
    
    async def _apply_evolution(self, conflict_id: int,
                             conflict: Dict[str, Any],
                             impact: Dict[str, Any]) -> Dict[str, Any]:
        """Apply evolution changes to conflict"""
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                results = []
                
                # Update progress
                if 'progress_change' in impact:
                    progress_result = await self.core.update_conflict_progress(
                        conflict_id, impact['progress_change']
                    )
                    results.append(('progress', progress_result))
                
                # Update stakeholders
                if 'stakeholder_changes' in impact:
                    for npc_id, changes in impact['stakeholder_changes'].items():
                        await self._update_stakeholder(
                            conn, conflict_id, int(npc_id), changes
                        )
                    results.append(('stakeholders', impact['stakeholder_changes']))
                
                # Add complications
                if 'complications' in impact:
                    for comp in impact['complications']:
                        await self._add_complication(conn, conflict_id, comp)
                    results.append(('complications', impact['complications']))
                
                # Update resolution paths
                if 'path_changes' in impact:
                    for path_id, changes in impact['path_changes'].items():
                        await self._update_resolution_path(
                            conn, conflict_id, path_id, changes
                        )
                    results.append(('paths', impact['path_changes']))
                
                return dict(results)
    
    async def _needs_natural_evolution(self, conflict: Dict[str, Any]) -> bool:
        """Check if conflict needs natural evolution"""
        # Don't evolve concluded conflicts
        if conflict['phase'] == ConflictPhase.CONCLUDED.value:
            return False
        
        # Check last update time
        async with get_db_connection_context() as conn:
            last_update = await conn.fetchval("""
                SELECT MAX(created_at) FROM ConflictMemoryEvents
                WHERE conflict_id = $1
            """, conflict['conflict_id'])
            
            if last_update:
                hours_since = (datetime.utcnow() - last_update).total_seconds() / 3600
                
                # Evolve based on phase
                if conflict['phase'] == ConflictPhase.BREWING.value:
                    return hours_since > 48
                elif conflict['phase'] == ConflictPhase.ACTIVE.value:
                    return hours_since > 24
                elif conflict['phase'] == ConflictPhase.CLIMAX.value:
                    return hours_since > 12
                    
        return True
    
    async def _get_conflict_age(self, conflict_id: int) -> int:
        """Get conflict age in days"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT start_day, 
                       (SELECT value FROM CurrentRoleplay 
                        WHERE key = 'CurrentDay' AND user_id = $2 
                        AND conversation_id = $3) as current_day
                FROM Conflicts
                WHERE conflict_id = $1
            """, conflict_id, self.user_id, self.conversation_id)
            
            if row:
                current = int(row['current_day']) if row['current_day'] else 1
                start = row['start_day']
                return current - start
            return 0
    
    async def _update_stakeholder(self, conn, conflict_id: int,
                                npc_id: int, changes: Dict[str, Any]):
        """Update stakeholder properties"""
        update_parts = []
        params = [conflict_id, npc_id]
        param_idx = 3
        
        for field, value in changes.items():
            if field in ['involvement_level', 'public_motivation', 
                        'leadership_ambition', 'faction_standing']:
                update_parts.append(f"{field} = ${param_idx}")
                params.append(value)
                param_idx += 1
        
        if update_parts:
            await conn.execute(f"""
                UPDATE ConflictStakeholders
                SET {', '.join(update_parts)}
                WHERE conflict_id = $1 AND npc_id = $2
            """, *params)
    
    async def _add_complication(self, conn, conflict_id: int,
                              complication: Dict[str, Any]):
        """Add a complication to conflict"""
        # Log as memory event
        await conn.execute("""
            INSERT INTO ConflictMemoryEvents
            (conflict_id, memory_text, significance, entity_type, entity_id)
            VALUES ($1, $2, $3, 'complication', $1)
        """,
        conflict_id,
        f"Complication: {complication['description']}",
        complication.get('significance', 6)
        )
        
        # If it's an internal conflict, create it
        if complication.get('type') == 'internal_faction':
            await conn.execute("""
                INSERT INTO InternalFactionConflicts
                (faction_id, conflict_name, description,
                 primary_npc_id, target_npc_id, prize,
                 approach, public_knowledge, current_phase,
                 progress, parent_conflict_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            complication['faction_id'], complication['name'],
            complication['description'], complication['primary_npc_id'],
            complication['target_npc_id'], complication['prize'],
            complication['approach'], False, 'brewing', 10, conflict_id
            )
    
    async def _update_resolution_path(self, conn, conflict_id: int,
                                    path_id: str, changes: Dict[str, Any]):
        """Update a resolution path"""
        if 'progress' in changes:
            await conn.execute("""
                UPDATE ResolutionPaths
                SET progress = LEAST(progress + $1, 100)
                WHERE conflict_id = $2 AND path_id = $3
            """, changes['progress'], conflict_id, path_id)
        
        if 'difficulty' in changes:
            await conn.execute("""
                UPDATE ResolutionPaths
                SET difficulty = $1
                WHERE conflict_id = $2 AND path_id = $3
            """, changes['difficulty'], conflict_id, path_id)
