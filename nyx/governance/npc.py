# nyx/governance/npc.py
"""
NPC behavior and relationship management.
"""
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from db.connection import get_db_connection_context
from agents import RunContextWrapper

logger = logging.getLogger(__name__)


class NPCGovernanceMixin:
    """Handles NPC-related governance functions."""
    
    async def modify_npc_behavior(self, npc_name: str, behavior_changes: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Modify an NPC's behavior through the LoreSystem.
        
        Args:
            npc_name: Name of the NPC (will be looked up)
            behavior_changes: Changes to apply to the NPC
            reason: Narrative reason for the change
        
        Returns:
            Result of the modification
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Modifying behavior for NPC {npc_name}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get NPC ID
        npc_id = await self._get_npc_id_by_name(npc_name)
        if not npc_id:
            logger.error(f"NPC '{npc_name}' not found. Cannot modify behavior.")
            return {"status": "error", "message": f"NPC '{npc_name}' not found"}
    
        # Delegate to LoreSystem
        result = await self.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCStats",
            entity_identifier={"npc_id": npc_id},
            updates=behavior_changes,
            reason=f"NPC behavior modification: {reason}"
        )
    
        return result

    async def assign_npc_to_location(self, npc_name: str, location_name: str, reason: str) -> Dict[str, Any]:
        """
        Assign an NPC to a new location through the LoreSystem.
        
        Args:
            npc_name: Name of the NPC
            location_name: Name of the location
            reason: Narrative reason for the move
            
        Returns:
            Result of the assignment
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Assigning {npc_name} to location {location_name}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get NPC ID
        npc_id = await self._get_npc_id_by_name(npc_name)
        if not npc_id:
            # Create the NPC if needed
            from lore.core import canon
            async with get_db_connection_context() as conn:
                npc_id = await canon.find_or_create_npc(
                    ctx, conn,
                    npc_name=npc_name,
                    role="Citizen"
                )
    
        # Ensure location exists
        from lore.core import canon
        async with get_db_connection_context() as conn:
            await canon.find_or_create_location(ctx, conn, location_name)
    
        # Update NPC's current location
        result = await self.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCStats",
            entity_identifier={"npc_id": npc_id},
            updates={"current_location": location_name},
            reason=f"Location assignment: {reason}"
        )
    
        return result

    async def create_npc_relationship(self, npc1_name: str, npc2_name: str, 
                                    relationship_type: str, details: Dict[str, Any], 
                                    reason: str) -> Dict[str, Any]:
        """
        Create or update a relationship between two NPCs.
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Creating relationship between {npc1_name} and {npc2_name}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get NPC IDs
        npc1_id = await self._get_npc_id_by_name(npc1_name)
        npc2_id = await self._get_npc_id_by_name(npc2_name)
    
        if not npc1_id or not npc2_id:
            missing = []
            if not npc1_id:
                missing.append(npc1_name)
            if not npc2_id:
                missing.append(npc2_name)
            return {"status": "error", "message": f"NPCs not found: {', '.join(missing)}"}
    
        # Create or update the social link
        async with get_db_connection_context() as conn:
            # First create/update the SocialLinks entry
            link_id = await conn.fetchval("""
                INSERT INTO SocialLinks (
                    user_id, conversation_id,
                    entity1_type, entity1_id, entity2_type, entity2_id,
                    link_type, link_level, link_history, dynamics,
                    relationship_stage
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11)
                ON CONFLICT (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
                DO UPDATE SET 
                    link_type = EXCLUDED.link_type,
                    link_level = EXCLUDED.link_level,
                    dynamics = EXCLUDED.dynamics,
                    relationship_stage = EXCLUDED.relationship_stage
                RETURNING link_id
            """, 
                self.user_id, self.conversation_id,
                'npc', npc1_id, 'npc', npc2_id,
                relationship_type, details.get('link_level', 50),
                json.dumps([{"timestamp": datetime.now().isoformat(), "event": reason}]),
                json.dumps(details), details.get('stage', 'acquaintance')
            )
            
            # Update NPCs' relationships fields
            relationships = {}
            for npc_id, other_name in [(npc1_id, npc2_name), (npc2_id, npc1_name)]:
                # Get current relationships
                current_rels = await conn.fetchval("""
                    SELECT relationships FROM NPCStats WHERE npc_id = $1
                """, npc_id)
                
                relationships = json.loads(current_rels) if current_rels else {}
                relationships[other_name] = {
                    "type": relationship_type,
                    "level": details.get("link_level", 50),
                    "details": details
                }
                
                # Update through LoreSystem (outside the connection context)
                await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCStats",
                    entity_identifier={"npc_id": npc_id},
                    updates={"relationships": relationships},
                    reason=f"Relationship update: {reason}"
                )
    
        return {"status": "committed", "relationship_created": True}

    async def _get_npc_id_by_name(self, npc_name: str) -> Optional[int]:
        """Retrieve an NPC ID by name from NPCStats."""
        try:
            async with get_db_connection_context() as conn:
                result = await conn.fetchval("""
                    SELECT npc_id FROM NPCStats 
                    WHERE npc_name = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_name, self.user_id, self.conversation_id)
                return result
        except Exception as e:
            logger.error(f"Error retrieving NPC ID for '{npc_name}': {e}")
            return None

    async def _get_npc_stat(self, npc_id: int, stat_name: str, default: int = 50) -> int:
        """Get a specific stat value for an NPC."""
        try:
            async with get_db_connection_context() as conn:
                value = await conn.fetchval(f"""
                    SELECT {stat_name} FROM NPCStats 
                    WHERE npc_id = $1
                """, npc_id)
                return value if value is not None else default
        except Exception as e:
            logger.error(f"Error getting NPC stat {stat_name}: {e}")
            return default
    
    async def _add_personality_trait(self, npc_id: int, new_trait: str) -> List[str]:
        """Add a personality trait to an NPC if they don't already have it."""
        try:
            async with get_db_connection_context() as conn:
                current_traits = await conn.fetchval("""
                    SELECT personality_traits FROM NPCStats 
                    WHERE npc_id = $1
                """, npc_id)
                
                traits = json.loads(current_traits) if current_traits else []
                if new_trait not in traits:
                    traits.append(new_trait)
                
                return traits
        except Exception as e:
            logger.error(f"Error adding personality trait: {e}")
            return [new_trait]

    async def _aligns_with_motivation(self, action_type: str, action_details: Dict[str, Any], 
                                     character_state: Dict[str, Any]) -> bool:
        """Check if an action aligns with character motivations using actual character data."""
        try:
            async with get_db_connection_context() as conn:
                # Get player's current stats and state
                player_stats = await conn.fetchrow("""
                    SELECT * FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                
                if not player_stats:
                    return True  # No data, allow action
                
                # Get player's relationships
                relationships = await conn.fetch("""
                    SELECT sl.*, ns.npc_name
                    FROM SocialLinks sl
                    JOIN NPCStats ns ON sl.entity2_id = ns.npc_id
                    WHERE sl.user_id = $1 AND sl.conversation_id = $2
                    AND sl.entity1_type = 'player' AND sl.entity2_type = 'npc'
                """, self.user_id, self.conversation_id)
                
                # Build relationship map
                rel_map = {r['npc_name']: r for r in relationships}
                
                # Check action against character state
                if action_type == 'betray':
                    target = action_details.get('target', '')
                    if target in rel_map:
                        # High trust/closeness makes betrayal unlikely
                        rel = rel_map[target]
                        if rel['link_level'] > 75:  # Strong positive relationship
                            return False
                            
                elif action_type == 'steal':
                    # Check if aligns with corruption level
                    if player_stats['corruption'] < 30:  # Low corruption
                        return False
                        
                elif action_type == 'help_selflessly':
                    # Check if aligns with personality
                    if player_stats['corruption'] > 70:  # High corruption
                        return False
                
                # Check against active addictions
                if action_type == 'resist_temptation':
                    target_npc = action_details.get('source', '')
                    addiction = await conn.fetchrow("""
                        SELECT level FROM PlayerAddictions
                        WHERE user_id = $1 AND conversation_id = $2
                        AND player_name = 'Chase' AND target_npc_id = (
                            SELECT npc_id FROM NPCStats 
                            WHERE npc_name = $3 AND user_id = $1 AND conversation_id = $2
                        )
                    """, self.user_id, self.conversation_id, target_npc)
                    
                    if addiction and addiction['level'] > 3:  # High addiction
                        return False
                        
            return True
        except Exception as e:
            logger.error(f"Error checking motivation alignment: {e}")
            return True

    def _disrupts_development(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        character_state: Dict[str, Any]
    ) -> bool:
        """Check if an action would disrupt character development."""
        # Get character development trajectory
        development = character_state.get("development", {})
        current_arc = development.get("current_arc", {})
        
        # Check if action would skip development
        if action_type == "skip_development":
            return True
            
        # Check if action would contradict development
        if action_type == "contradict_development":
            return True
            
        return False

    def _maintains_relationships(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        character_state: Dict[str, Any]
    ) -> bool:
        """Check if an action maintains relationship consistency."""
        # Get relationship dynamics
        relationships = character_state.get("relationships", {})
        
        # Check if action would break relationships
        if action_type == "break_relationship":
            return False
            
        # Check if action maintains relationship dynamics
        if action_type == "maintain_relationship":
            return True
            
        return True
