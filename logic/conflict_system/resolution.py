# logic/conflict_system/resolution.py
"""
Sophisticated conflict resolution system
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple

from agents import Agent, Runner, ModelSettings
from db.connection import get_db_connection_context
from lore.core import canon
from logic.resource_management import ResourceManager

from .core import ConflictCore, ConflictPhase, ResolutionApproach

logger = logging.getLogger(__name__)

# Resolution Strategy Agent
resolution_agent = Agent(
    name="Resolution Strategy Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.7),
    instructions="""
    You develop and execute conflict resolution strategies.
    
    Consider:
    - Current conflict state and phase
    - Available resolution paths
    - Stakeholder positions
    - Player resources and involvement
    - Historical precedents
    
    Create strategies that:
    - Feel earned and meaningful
    - Have lasting consequences
    - Respect player choices
    - Create new story opportunities
    
    Output resolution plan as JSON.
    """
)

class ConflictResolver:
    """Advanced conflict resolution system"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.core = ConflictCore(user_id, conversation_id)
        self.resource_manager = ResourceManager(user_id, conversation_id)
    
    async def resolve_conflict(self, conflict_id: int,
                             resolution_method: str,
                             resolution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a conflict with sophisticated outcomes"""
        # Get conflict details
        conflict = await self.core.get_conflict_details(conflict_id)
        if not conflict:
            return {"error": "Conflict not found"}
        
        # Validate resolution readiness
        if not await self._can_resolve(conflict):
            return {"error": "Conflict not ready for resolution", 
                   "reason": self._get_resolution_blocker(conflict)}
        
        # Generate resolution strategy
        strategy = await self._generate_resolution_strategy(
            conflict, resolution_method, resolution_data
        )
        
        # Execute resolution
        result = await self._execute_resolution(conflict_id, conflict, strategy)
        
        # Generate and apply consequences
        consequences = await self._generate_consequences(conflict, result)
        await self._apply_consequences(consequences)
        
        return {
            "success": True,
            "conflict_id": conflict_id,
            "resolution": result,
            "consequences": consequences
        }
    
    async def track_story_beat(self, conflict_id: int, path_id: str,
                             beat_description: str,
                             involved_npcs: List[int],
                             progress_value: float) -> Dict[str, Any]:
        """Track a story beat that advances a resolution path"""
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Record the beat
                beat_id = await conn.fetchval("""
                    INSERT INTO PathStoryBeats
                    (conflict_id, path_id, description, involved_npcs, 
                     progress_value, created_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    RETURNING beat_id
                """, conflict_id, path_id, beat_description,
                json.dumps(involved_npcs), progress_value)
                
                # Update path progress
                path = await conn.fetchrow("""
                    SELECT progress, is_completed FROM ResolutionPaths
                    WHERE conflict_id = $1 AND path_id = $2
                """, conflict_id, path_id)
                
                if not path:
                    return {"error": "Path not found"}
                
                new_progress = min(100, path['progress'] + progress_value)
                is_completed = new_progress >= 100
                
                await conn.execute("""
                    UPDATE ResolutionPaths
                    SET progress = $1, is_completed = $2,
                        completion_date = CASE 
                            WHEN $2 THEN NOW() 
                            ELSE completion_date 
                        END
                    WHERE conflict_id = $3 AND path_id = $4
                """, new_progress, is_completed, conflict_id, path_id)
                
                # Check if this triggers conflict advancement
                if is_completed:
                    await self._check_conflict_completion(conn, conflict_id)
                
                return {
                    "beat_id": beat_id,
                    "path_id": path_id,
                    "description": beat_description,
                    "progress_value": progress_value,
                    "new_progress": new_progress,
                    "is_completed": is_completed
                }
    
    async def _can_resolve(self, conflict: Dict[str, Any]) -> bool:
        """Check if conflict can be resolved"""
        # Must be in resolution phase or have high progress
        if conflict['phase'] not in [ConflictPhase.RESOLUTION.value, 
                                     ConflictPhase.CLIMAX.value]:
            return conflict['progress'] >= 80
        
        # At least one path should have progress
        paths = conflict.get('resolution_paths', [])
        return any(p['progress'] > 50 for p in paths)
    
    def _get_resolution_blocker(self, conflict: Dict[str, Any]) -> str:
        """Get reason why conflict can't be resolved"""
        if conflict['progress'] < 50:
            return "Conflict hasn't progressed enough"
        
        paths = conflict.get('resolution_paths', [])
        if not any(p['progress'] > 0 for p in paths):
            return "No resolution paths have been pursued"
        
        return "Conflict needs more development"
    
    async def _generate_resolution_strategy(self, conflict: Dict[str, Any],
                                          method: str,
                                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a resolution strategy"""
        # Get completed paths
        completed_paths = [p for p in conflict['resolution_paths'] 
                          if p['is_completed']]
        active_paths = [p for p in conflict['resolution_paths']
                       if p['progress'] > 25 and not p['is_completed']]
        
        prompt = f"""
        Generate resolution strategy for this conflict:
        
        Conflict: {conflict['conflict_name']}
        Type: {conflict['conflict_type']}
        Phase: {conflict['phase']} ({conflict['progress']}%)
        
        Resolution Method: {method}
        Player Data: {json.dumps(data, indent=2)}
        
        Completed Paths: {json.dumps(completed_paths, indent=2)}
        Active Paths: {json.dumps(active_paths, indent=2)}
        
        Key Stakeholders:
        {json.dumps([{
            'name': s['npc_name'],
            'public_stance': s['public_motivation'],
            'private_goal': s['private_motivation'],
            'involvement': s['involvement_level']
        } for s in conflict['stakeholders'][:5]], indent=2)}
        
        Create a resolution that:
        1. Feels earned based on paths taken
        2. Addresses stakeholder goals
        3. Has meaningful consequences
        4. Sets up future story potential
        
        Output complete resolution details.
        """
        
        result = await Runner.run(
            resolution_agent,
            prompt,
            self.core.ctx
        )
        
        return json.loads(result.final_output)
    
    async def _execute_resolution(self, conflict_id: int,
                                conflict: Dict[str, Any],
                                strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the resolution strategy"""
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Update conflict status
                await conn.execute("""
                    UPDATE Conflicts
                    SET is_active = FALSE, 
                        phase = $1,
                        progress = 100,
                        outcome = $2,
                        resolution_description = $3,
                        resolved_at = NOW()
                    WHERE conflict_id = $4
                """, ConflictPhase.CONCLUDED.value,
                strategy['outcome_type'],
                strategy['description'],
                conflict_id)
                
                # Update stakeholder outcomes
                for stakeholder_outcome in strategy.get('stakeholder_outcomes', []):
                    await self._update_stakeholder_outcome(
                        conn, conflict_id, stakeholder_outcome
                    )
                
                # Create resolution record
                resolution_id = await conn.fetchval("""
                    INSERT INTO ConflictResolutions
                    (conflict_id, resolution_type, winning_faction,
                     player_role, narrative_summary, canonical_changes)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING resolution_id
                """,
                conflict_id, strategy['resolution_type'],
                strategy.get('winning_faction'),
                strategy.get('player_role', 'participant'),
                strategy['narrative_summary'],
                json.dumps(strategy.get('canonical_changes', []))
                )
                
                # Log canonical event
                await canon.log_canonical_event(
                    self.core.ctx.context, conn,
                    f"Conflict resolved: {conflict['conflict_name']} - {strategy['outcome_type']}",
                    tags=["conflict", "resolution", strategy['resolution_type']],
                    significance=8
                )
                
                return {
                    "resolution_id": resolution_id,
                    "outcome_type": strategy['outcome_type'],
                    "description": strategy['description'],
                    "narrative_summary": strategy['narrative_summary']
                }
    
    async def _generate_consequences(self, conflict: Dict[str, Any],
                                   resolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate consequences based on resolution"""
        consequences = []
        
        # Player stat changes
        if conflict['player_involvement']:
            involvement = conflict['player_involvement']['involvement_level']
            if involvement in ['participating', 'leading']:
                consequences.extend(await self._generate_player_consequences(
                    conflict, resolution, involvement
                ))
        
        # World changes
        consequences.extend(await self._generate_world_consequences(
            conflict, resolution
        ))
        
        # Relationship changes
        consequences.extend(await self._generate_relationship_consequences(
            conflict, resolution
        ))
        
        # Item/perk rewards
        consequences.extend(await self._generate_rewards(
            conflict, resolution
        ))
        
        return consequences
    
    async def _generate_player_consequences(self, conflict: Dict[str, Any],
                                          resolution: Dict[str, Any],
                                          involvement: str) -> List[Dict[str, Any]]:
        """Generate player-specific consequences"""
        consequences = []
        
        # Base stat changes
        if resolution['outcome_type'] == 'victory':
            stat_changes = {
                'confidence': 3 if involvement == 'leading' else 2,
                'mental_resilience': 2
            }
        elif resolution['outcome_type'] == 'compromise':
            stat_changes = {
                'willpower': 2,
                'mental_resilience': 1
            }
        else:  # defeat
            stat_changes = {
                'confidence': -1,
                'mental_resilience': 3,
                'obedience': 2
            }
        
        consequences.append({
            'type': 'player_stats',
            'changes': stat_changes,
            'description': self._describe_stat_changes(stat_changes)
        })
        
        return consequences
    
    async def _generate_world_consequences(self, conflict: Dict[str, Any],
                                         resolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate world-state consequences"""
        scale_multiplier = {
            'personal': 1,
            'local': 2,
            'regional': 3,
            'world': 4
        }.get(conflict.get('scale', 'local'), 2)
        
        return [{
            'type': 'world_state',
            'impact_level': scale_multiplier,
            'changes': resolution.get('world_changes', []),
            'description': f"The resolution of {conflict['conflict_name']} has shifted the balance of power"
        }]
    
    async def _generate_relationship_consequences(self, conflict: Dict[str, Any],
                                                resolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate relationship changes"""
        consequences = []
        
        for stakeholder in conflict['stakeholders']:
            if stakeholder['npc_id'] in resolution.get('winning_npcs', []):
                change = 10
            elif stakeholder['npc_id'] in resolution.get('losing_npcs', []):
                change = -10
            else:
                change = 0
            
            if change != 0:
                consequences.append({
                    'type': 'relationship',
                    'npc_id': stakeholder['npc_id'],
                    'npc_name': stakeholder['npc_name'],
                    'change': change
                })
        
        return consequences
    
    async def _generate_rewards(self, conflict: Dict[str, Any],
                              resolution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate item and perk rewards"""
        rewards = []
        
        # Determine reward tier based on conflict scale
        tier = {
            'personal': 1,
            'local': 2,
            'regional': 3,
            'world': 4
        }.get(conflict.get('scale', 'local'), 2)
        
        # Item reward
        primary_path = next((p for p in conflict['resolution_paths'] 
                           if p['is_completed']), None)
        if primary_path:
            approach = primary_path.get('approach_type', 'neutral')
            item = self._generate_item_reward(approach, tier)
            rewards.append({
                'type': 'item',
                'item': item,
                'description': f"Received {item['name']} for resolving the conflict"
            })
        
        # Perk reward for major conflicts
        if tier >= 3:
            perk = self._generate_perk_reward(
                resolution.get('resolution_type', 'standard'), tier
            )
            rewards.append({
                'type': 'perk',
                'perk': perk,
                'description': f"Gained perk: {perk['name']}"
            })
        
        return rewards
    
    def _generate_item_reward(self, approach: str, tier: int) -> Dict[str, Any]:
        """Generate an appropriate item reward"""
        items_by_approach = {
            'diplomatic': [
                {"name": "Negotiator's Seal", "tier": 1},
                {"name": "Treaty Pendant", "tier": 2},
                {"name": "Peacekeeper's Regalia", "tier": 3},
                {"name": "Crown of Accord", "tier": 4}
            ],
            'forceful': [
                {"name": "Victor's Trophy", "tier": 1},
                {"name": "Conqueror's Brand", "tier": 2},
                {"name": "Dominator's Scepter", "tier": 3},
                {"name": "Throne of Subjugation", "tier": 4}
            ],
            'manipulative': [
                {"name": "Shadow Veil", "tier": 1},
                {"name": "Puppeteer's Strings", "tier": 2},
                {"name": "Mastermind's Codex", "tier": 3},
                {"name": "Web of Lies", "tier": 4}
            ]
        }
        
        items = items_by_approach.get(approach, items_by_approach['diplomatic'])
        item = next((i for i in items if i['tier'] == tier), items[0])
        
        return {
            "name": item['name'],
            "description": f"A {approach} resolution reward",
            "tier": tier,
            "category": "conflict_reward"
        }
    
    def _generate_perk_reward(self, resolution_type: str, tier: int) -> Dict[str, Any]:
        """Generate an appropriate perk reward"""
        return {
            "name": f"{resolution_type.title()} Resolver",
            "description": f"Expertise in {resolution_type} conflict resolution",
            "tier": tier - 2,  # Perks are tier 1-2
            "effects": [f"+{tier} to {resolution_type} approaches"]
        }
    
    async def _apply_consequences(self, consequences: List[Dict[str, Any]]):
        """Apply all consequences"""
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                for consequence in consequences:
                    if consequence['type'] == 'player_stats':
                        await self._apply_stat_changes(conn, consequence['changes'])
                    elif consequence['type'] == 'relationship':
                        await self._apply_relationship_change(conn, consequence)
                    elif consequence['type'] == 'item':
                        await self._grant_item(conn, consequence['item'])
                    elif consequence['type'] == 'perk':
                        await self._grant_perk(conn, consequence['perk'])
    
    async def _update_stakeholder_outcome(self, conn, conflict_id: int,
                                        outcome: Dict[str, Any]):
        """Update stakeholder final status"""
        await conn.execute("""
            UPDATE ConflictStakeholders
            SET final_outcome = $1, satisfaction_level = $2
            WHERE conflict_id = $3 AND npc_id = $4
        """, outcome['result'], outcome['satisfaction'],
        conflict_id, outcome['npc_id'])
    
    async def _check_conflict_completion(self, conn, conflict_id: int):
        """Check if conflict should auto-complete"""
        # Count completed paths
        completed = await conn.fetchval("""
            SELECT COUNT(*) FROM ResolutionPaths
            WHERE conflict_id = $1 AND is_completed = TRUE
        """, conflict_id)
        
        if completed >= 2:  # Multiple paths completed
            await self.core.update_conflict_progress(conflict_id, 20)
    
    def _describe_stat_changes(self, changes: Dict[str, int]) -> str:
        """Create narrative description of stat changes"""
        descriptions = []
        for stat, value in changes.items():
            if value > 0:
                descriptions.append(f"Your {stat} increased by {value}")
            elif value < 0:
                descriptions.append(f"Your {stat} decreased by {abs(value)}")
        
        return ". ".join(descriptions)
    
    async def _apply_stat_changes(self, conn, changes: Dict[str, int]):
        """Apply stat changes to player"""
        for stat, value in changes.items():
            await conn.execute(f"""
                UPDATE PlayerStats
                SET {stat} = GREATEST(0, LEAST(100, {stat} + $1))
                WHERE user_id = $2 AND conversation_id = $3
            """, value, self.user_id, self.conversation_id)
    
    async def _apply_relationship_change(self, conn, consequence: Dict[str, Any]):
        """Apply relationship changes"""
        # This would integrate with your relationship system
        pass
    
    async def _grant_item(self, conn, item: Dict[str, Any]):
        """Grant item to player"""
        await canon.find_or_create_inventory_item(
            self.core.ctx.context, conn,
            item_name=item['name'],
            player_name="Chase",
            item_description=item['description'],
            item_category=item['category'],
            item_properties={"tier": item['tier']},
            quantity=1,
            equipped=False
        )
    
    async def _grant_perk(self, conn, perk: Dict[str, Any]):
        """Grant perk to player"""
        # This would integrate with your perk system
        pass
