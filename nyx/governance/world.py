# nyx/governance/world.py
"""
World state and setting management.
"""
import logging
import json
import re
from typing import Dict, Any, Optional, List
from db.connection import get_db_connection_context
from agents import RunContextWrapper

logger = logging.getLogger(__name__)


class WorldGovernanceMixin:
    """Handles world state and setting governance functions."""
    
    async def evolve_world_state(self, evolution_type: str, parameters: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Evolve the world state in a specific way through the LoreSystem.
        
        Args:
            evolution_type: Type of evolution (e.g., 'cultural_shift', 'technological_advance')
            parameters: Parameters for the evolution
            reason: Narrative reason for the evolution
        
        Returns:
            Result of the evolution
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"NYX: Evolving world state: {evolution_type}")

        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        
        results = []

        if evolution_type == "cultural_shift":
            # Cultural shifts might affect multiple entities
            affected_regions = parameters.get("affected_regions", [])
            cultural_changes = parameters.get("changes", {})
            
            for region_id in affected_regions:
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="GeographicRegions",
                    entity_identifier={"id": region_id},
                    updates={"cultural_traits": cultural_changes.get("new_traits", [])},
                    reason=f"Cultural shift in region: {reason}"
                )
                results.append(result)
                
        elif evolution_type == "technological_advance":
            # Technological advances might update multiple nations
            affected_nations = parameters.get("affected_nations", [])
            tech_level = parameters.get("technology_level", 5)
            
            for nation_id in affected_nations:
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="Nations",
                    entity_identifier={"id": nation_id},
                    updates={"technology_level": tech_level},
                    reason=f"Technological advancement: {reason}"
                )
                results.append(result)
                
        elif evolution_type == "economic_shift":
            # Economic shifts affect resources and trade
            affected_entities = parameters.get("affected_entities", [])
            economic_changes = parameters.get("changes", {})
            
            for entity in affected_entities:
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type=entity["type"],
                    entity_identifier={"id": entity["id"]},
                    updates=economic_changes,
                    reason=f"Economic shift: {reason}"
                )
                results.append(result)

        return {
            "status": "completed",
            "evolution_type": evolution_type,
            "results": results,
            "reason": reason
        }

    async def enact_political_change(self, nation_name: str, change_type: str, details: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Enact a political change in a nation through the LoreSystem.
        
        Args:
            nation_name: Name of the nation (will be looked up)
            change_type: Type of change (e.g., 'leadership', 'government', 'policy')
            details: Specific details of the change
            reason: Narrative reason for the change
        
        Returns:
            Result of the change enactment
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Enacting political change in nation {nation_name}: {change_type}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get nation ID
        nation_id = await self._get_nation_id_by_name(nation_name)
        if not nation_id:
            # Create the nation if it doesn't exist
            from lore.core import canon
            async with get_db_connection_context() as conn:
                nation_id = await canon.find_or_create_nation(
                    ctx, conn,
                    nation_name=nation_name,
                    government_type=details.get('government_type', 'Unknown')
                )
    
        # Map change_type to appropriate updates
        updates = {}
        if change_type == "leadership":
            if "new_leader_name" in details:
                # Get or create the new leader NPC
                new_leader_id = await self._get_npc_id_by_name(details["new_leader_name"])
                if not new_leader_id:
                    from lore.core import canon
                    async with get_db_connection_context() as conn:
                        new_leader_id = await canon.find_or_create_npc(
                            ctx, conn,
                            npc_name=details["new_leader_name"],
                            role="Political Leader",
                            affiliations=[nation_name]
                        )
                updates["leader_npc_id"] = new_leader_id
            if "leadership_structure" in details:
                updates["leadership_structure"] = details["leadership_structure"]
        elif change_type == "government":
            if "government_type" in details:
                updates["government_type"] = details["government_type"]
            if "matriarchy_level" in details:
                updates["matriarchy_level"] = details["matriarchy_level"]
        elif change_type == "policy":
            # For policies, we might need to update JSON fields
            if "diplomatic_stance" in details:
                updates["diplomatic_stance"] = details["diplomatic_stance"]
            if "economic_focus" in details:
                updates["economic_focus"] = details["economic_focus"]
    
        # Delegate to LoreSystem
        result = await self.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="nations",  # Use lowercase as per schema
            entity_identifier={"id": nation_id},
            updates=updates,
            reason=f"Political change ({change_type}): {reason}"
        )
    
        # Record this as a major narrative event
        if result.get("status") == "committed":
            await self._record_narrative_event(
                event_type="political_change",
                details={
                    "nation_id": nation_id,
                    "nation_name": nation_name,
                    "change_type": change_type,
                    "updates": updates,
                    "reason": reason
                }
            )
    
        return result

    async def update_faction_relations(self, faction_name: str, relation_updates: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Update faction relations through the LoreSystem.
        
        Args:
            faction_name: Name of the faction
            relation_updates: Updates to faction relations including:
                - ally_names: List of ally faction names
                - rival_names: List of rival faction names  
                - public_reputation: New reputation value
            reason: Narrative reason for the change
        
        Returns:
            Result of the update
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Updating relations for faction {faction_name}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get faction ID
        faction_id = await self._get_faction_id_by_name(faction_name)
        if not faction_id:
            logger.error(f"Faction '{faction_name}' not found.")
            return {"status": "error", "message": f"Faction '{faction_name}' not found"}
    
        # Convert faction names to IDs in relation updates
        updates = {}
        
        if "ally_names" in relation_updates:
            ally_ids = []
            for ally_name in relation_updates["ally_names"]:
                ally_id = await self._get_faction_id_by_name(ally_name)
                if ally_id:
                    ally_ids.append(ally_id)
            updates["allies"] = ally_ids
            
        if "rival_names" in relation_updates:
            rival_ids = []
            for rival_name in relation_updates["rival_names"]:
                rival_id = await self._get_faction_id_by_name(rival_name)
                if rival_id:
                    rival_ids.append(rival_id)
            updates["rivals"] = rival_ids
            
        if "public_reputation" in relation_updates:
            updates["public_reputation"] = relation_updates["public_reputation"]
    
        # Delegate to LoreSystem
        result = await self.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="Factions",
            entity_identifier={"id": faction_id},
            updates=updates,
            reason=f"Faction relations update: {reason}"
        )
    
        return result

    async def create_local_group(self, group_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Create a local group or organization (not necessarily political).
        
        Args:
            group_data: Data including:
                - name: Group name (e.g., "Book Club", "Local Band", "Parent Committee")
                - type: Type (e.g., "social", "hobby", "community", "educational")
                - scope: Scope (e.g., "school", "neighborhood", "online")
                - meeting_place: Where they meet
                - members: List of member names
                - activities: What they do
            reason: Why this group is being created
        """
        if not self._initialized:
            await self.initialize()
            
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        
        # Create the faction with appropriate type
        from lore.core import canon
        async with get_db_connection_context() as conn:
            faction_id = await canon.find_or_create_faction(
                ctx, conn,
                faction_name=group_data["name"],
                type=group_data.get("type", "social"),
                description=group_data.get("description", f"A {group_data.get('type', 'social')} group"),
                values=group_data.get("values", ["community", "shared interests"]),
                goals=group_data.get("goals", ["meet regularly", "enjoy activities"]),
                influence_scope=group_data.get("scope", "local"),
                power_level=2,  # Low power level for local groups
                territory=[group_data.get("meeting_place", "various locations")]
            )
        
        # Add members as allies/affiliates
        if "members" in group_data:
            member_ids = []
            for member_name in group_data["members"]:
                npc_id = await self._get_npc_id_by_name(member_name)
                if npc_id:
                    member_ids.append(npc_id)
                    # Update NPC's affiliations
                    await self.lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="NPCStats", 
                        entity_identifier={"npc_id": npc_id},
                        updates={"affiliations": [group_data["name"]]},
                        reason=f"Joined {group_data['name']}"
                    )
        
        return {"status": "success", "faction_id": faction_id, "type": "local_group"}

    async def _get_faction_id_by_name(self, faction_name: str) -> Optional[int]:
        """
        Retrieve a faction ID by name from the Factions table.
        """
        try:
            async with get_db_connection_context() as conn:
                result = await conn.fetchval("""
                    SELECT id FROM Factions 
                    WHERE name = $1 AND user_id = $2 AND conversation_id = $3
                """, faction_name, self.user_id, self.conversation_id)
                return result
        except Exception as e:
            logger.error(f"Error retrieving faction ID for '{faction_name}': {e}")
            return None

    async def _get_nation_id_by_name(self, nation_name: str) -> Optional[int]:
        """Retrieve a nation ID by name from the Nations table."""
        try:
            async with get_db_connection_context() as conn:
                result = await conn.fetchval("""
                    SELECT id FROM Nations 
                    WHERE LOWER(name) = LOWER($1)
                """, nation_name)
                return result
        except Exception as e:
            logger.error(f"Error retrieving nation ID for '{nation_name}': {e}")
            return None

    async def _violates_world_rules(self, action_type: str, action_details: Dict[str, Any], 
                                   world_state: Dict[str, Any]) -> bool:
        """Check if an action violates established world rules using GameRules table."""
        try:
            async with get_db_connection_context() as conn:
                # Get all active game rules
                rules = await conn.fetch("""
                    SELECT rule_name, condition, effect
                    FROM GameRules
                """)
                
                for rule in rules:
                    condition = rule['condition'].lower()
                    effect = rule['effect'].lower()
                    
                    # Parse conditions and check against action
                    # This is a simplified version - you might want more complex parsing
                    if action_type.lower() in condition:
                        if 'prohibited' in effect or 'forbidden' in effect or 'cannot' in effect:
                            return True
                            
                    # Check stat-based rules
                    if 'stat:' in condition:
                        # Extract stat requirements
                        stat_match = re.search(r'stat:(\w+)\s*([<>=]+)\s*(\d+)', condition)
                        if stat_match:
                            stat_name = stat_match.group(1)
                            operator = stat_match.group(2)
                            value = int(stat_match.group(3))
                            
                            # Get player's current stat
                            player_stat = await conn.fetchval(f"""
                                SELECT {stat_name} FROM PlayerStats
                                WHERE user_id = $1 AND conversation_id = $2 
                                AND player_name = 'Chase'
                                ORDER BY timestamp DESC
                                LIMIT 1
                            """, self.user_id, self.conversation_id)
                            
                            if player_stat is not None:
                                if operator == '<' and player_stat < value:
                                    if action_type in effect:
                                        return True
                                elif operator == '>' and player_stat > value:
                                    if action_type in effect:
                                        return True
                                        
                # Check location-based restrictions
                if 'location' in action_details:
                    location = action_details['location']
                    location_data = await conn.fetchrow("""
                        SELECT access_restrictions, local_customs
                        FROM Locations
                        WHERE user_id = $1 AND conversation_id = $2
                        AND location_name = $3
                    """, self.user_id, self.conversation_id, location)
                    
                    if location_data:
                        restrictions = location_data['access_restrictions'] or []
                        customs = location_data['local_customs'] or []
                        
                        # Check if action violates local customs
                        for custom in customs:
                            if action_type in custom.lower():
                                return True
                                
            return False
        except Exception as e:
            logger.error(f"Error checking world rule violations: {e}")
            return False

    def _maintains_logical_consistency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> bool:
        """Check if an action maintains logical consistency."""
        # Get world logic and causality
        logic = world_state.get("logic", {})
        causality = world_state.get("causality", {})
        
        # Check if action breaks logic
        if action_type == "break_logic":
            return False
            
        # Check if action maintains causality
        if action_type == "maintain_causality":
            return True
            
        return True

    def _maintains_lore_consistency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> bool:
        """Check if an action maintains established lore consistency."""
        # Get established lore and history
        lore = world_state.get("lore", {})
        history = world_state.get("history", {})
        
        # Check if action contradicts lore
        if action_type == "contradict_lore":
            return False
            
        # Check if action maintains history
        if action_type == "maintain_history":
            return True
            
        return True

    async def _calculate_world_integrity(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> float:
        """Calculate how well an action maintains world integrity."""
        impact_score = 0.0
        
        # Check for world rule violations
        if await self._violates_world_rules(action_type, action_details, world_state):
            impact_score += 0.5
        
        # Check for logical consistency
        if not self._maintains_logical_consistency(action_type, action_details, world_state):
            impact_score += 0.3
        
        # Check for established lore consistency
        if not self._maintains_lore_consistency(action_type, action_details, world_state):
            impact_score += 0.2
        
        return min(1.0, impact_score)

    async def _suggest_world_alternative(self, world_state: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alternative that respects world rules."""
        setting = world_state.get("setting", "Unknown")
        rules = world_state.get("rules", {})
        
        return {
            "type": "world_alternative",
            "suggestion": f"Work within the established rules of {setting}",
            "specific_options": [
                "Use existing game mechanics to achieve your goal",
                "Find creative solutions within world constraints",
                "Seek help from factions or NPCs with relevant expertise"
            ],
            "reasoning": "Respecting world consistency enhances immersion"
        }

    async def _has_required_resources(self, action_details: Dict[str, Any], 
                                     requirements: Dict[str, Any]) -> bool:
        """Check if player has resources required for an action."""
        try:
            async with get_db_connection_context() as conn:
                # Get player's current resources
                resources = await conn.fetchrow("""
                    SELECT money, supplies, influence
                    FROM PlayerResources
                    WHERE user_id = $1 AND conversation_id = $2 
                    AND player_name = 'Chase'
                """, self.user_id, self.conversation_id)
                
                if not resources:
                    return False
                    
                # Check each requirement
                required = action_details.get('requirements', {})
                
                if 'money' in required and resources['money'] < required['money']:
                    return False
                if 'supplies' in required and resources['supplies'] < required['supplies']:
                    return False  
                if 'influence' in required and resources['influence'] < required['influence']:
                    return False
                    
                # Check inventory requirements
                if 'items' in required:
                    for item_name in required['items']:
                        has_item = await conn.fetchval("""
                            SELECT COUNT(*) FROM PlayerInventory
                            WHERE user_id = $1 AND conversation_id = $2
                            AND player_name = 'Chase' AND item_name = $3
                            AND quantity > 0
                        """, self.user_id, self.conversation_id, item_name)
                        
                        if not has_item:
                            return False
                            
                # Check perk requirements
                if 'perks' in required:
                    for perk_name in required['perks']:
                        has_perk = await conn.fetchval("""
                            SELECT COUNT(*) FROM PlayerPerks
                            WHERE user_id = $1 AND conversation_id = $2
                            AND perk_name = $3
                        """, self.user_id, self.conversation_id, perk_name)
                        
                        if not has_perk:
                            return False
                            
            return True
        except Exception as e:
            logger.error(f"Error checking resource requirements: {e}")
            return True  # Default to allowing if check fails
