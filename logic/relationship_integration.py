# logic/relationship_integration.py

"""
Relationship integration module connecting the enhanced relationship system 
from IntegratedNPCSystem with the existing game social links.
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import asyncpg

from db.connection import get_db_connection_context
from logic.fully_integrated_npc_system import IntegratedNPCSystem
from lore.core import canon
from lore.lore_system import LoreSystem

logger = logging.getLogger(__name__)

class RelationshipIntegration:
    """
    Bridge class that integrates the sophisticated relationship management from 
    IntegratedNPCSystem with the existing social links handling.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize relationship integration.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_system = IntegratedNPCSystem(user_id, conversation_id)
        self.ctx = type('Context', (), {'user_id': user_id, 'conversation_id': conversation_id})()
    
    async def create_relationship(self, entity1_type: str, entity1_id: int,
                                entity2_type: str, entity2_id: int,
                                relationship_type: str = None, 
                                initial_level: int = 0) -> Dict[str, Any]:
        """
        Create a new relationship between two entities.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            relationship_type: Type of relationship
            initial_level: Initial relationship level
            
        Returns:
            Dictionary with relationship data
        """
        return await self.npc_system.create_relationship(
            entity1_type, entity1_id,
            entity2_type, entity2_id,
            relationship_type, initial_level
        )
    
    async def get_relationship(self, entity1_type: str, entity1_id: int,
                             entity2_type: str, entity2_id: int) -> Optional[Dict[str, Any]]:
        """
        Get relationship data between two entities.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            
        Returns:
            Dictionary with relationship data or None if not found
        """
        return await self.npc_system.get_relationship(
            entity1_type, entity1_id,
            entity2_type, entity2_id
        )
    
    async def update_dimensions(self, link_id: int, 
                              dimension_changes: Dict[str, int],
                              reason: str = None) -> Dict[str, Any]:
        """
        Update specific dimensions of a relationship.
        
        Args:
            link_id: ID of the relationship link
            dimension_changes: Dictionary of dimension changes
            reason: Reason for the changes
            
        Returns:
            Dictionary with update results
        """
        return await self.npc_system.update_relationship_dimensions(
            link_id, dimension_changes, reason
        )
    
    async def increase_tension(self, link_id: int, 
                             amount: int, 
                             reason: str = None) -> Dict[str, Any]:
        """
        Increase tension in a relationship.
        
        Args:
            link_id: ID of the relationship link
            amount: Amount to increase tension by
            reason: Reason for the tension increase
            
        Returns:
            Dictionary with update results
        """
        return await self.npc_system.increase_relationship_tension(
            link_id, amount, reason
        )
    
    async def release_tension(self, link_id: int,
                            amount: int,
                            resolution_type: str = "positive",
                            reason: str = None) -> Dict[str, Any]:
        """
        Release tension in a relationship.
        
        Args:
            link_id: ID of the relationship link
            amount: Amount to decrease tension by
            resolution_type: Type of resolution
            reason: Reason for the tension release
            
        Returns:
            Dictionary with update results
        """
        return await self.npc_system.release_relationship_tension(
            link_id, amount, resolution_type, reason
        )
    
    async def check_for_events(self) -> List[Dict[str, Any]]:
        """
        Check for significant relationship events.
        
        Returns:
            List of event dictionaries
        """
        return await self.npc_system.check_for_relationship_events()
    
    async def apply_crossroads_choice(self, link_id: int,
                                    crossroads_name: str,
                                    choice_index: int) -> Dict[str, Any]:
        """
        Apply a choice in a relationship crossroads.
        
        Args:
            link_id: ID of the relationship link
            crossroads_name: Name of the crossroads
            choice_index: Index of the selected choice
            
        Returns:
            Dictionary with the results
        """
        return await self.npc_system.apply_crossroads_choice(
            link_id, crossroads_name, choice_index
        )
    
    async def add_link_event(self, link_id: int, event_text: str) -> bool:
        """
        Add an event to a relationship's history.
        
        Args:
            link_id: ID of the relationship link
            event_text: Text describing the event
            
        Returns:
            True if successful
        """
        return await self.npc_system.add_event_to_link(link_id, event_text)
    
    async def get_dynamic_level(self, entity1_type: str, entity1_id: int,
                              entity2_type: str, entity2_id: int,
                              dynamic_name: str) -> int:
        """
        Get the level of a specific relationship dynamic.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            dynamic_name: Name of the dynamic
            
        Returns:
            Current level of the dynamic
        """
        return await self.npc_system.get_dynamic_level(
            entity1_type, entity1_id, entity2_type, entity2_id, dynamic_name
        )
    
    async def update_dynamic(self, entity1_type: str, entity1_id: int,
                           entity2_type: str, entity2_id: int,
                           dynamic_name: str, change: int) -> int:
        """
        Update a specific relationship dynamic.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            dynamic_name: Name of the dynamic
            change: Amount to change the dynamic by
            
        Returns:
            New level of the dynamic
        """
        return await self.npc_system.update_dynamic(
            entity1_type, entity1_id, entity2_type, entity2_id, dynamic_name, change
        )
    
    async def create_npc_group(self, name: str, description: str, 
                             member_ids: List[int]) -> Dict[str, Any]:
        """
        Create a group of NPCs.
        
        Args:
            name: Name of the group
            description: Description of the group
            member_ids: List of NPC IDs to include
            
        Returns:
            Group information
        """
        return await self.npc_system.create_npc_group(name, description, member_ids)
    
    async def add_npc_to_group(self, group_id: int, npc_id: int, role: str = "member") -> bool:
        """
        Asynchronously add an NPC to a group.

        Args:
            group_id: ID of the group
            npc_id: ID of the NPC
            role: Role of the NPC in the group

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with get_db_connection_context() as conn:
                # Use fetchrow which returns a Record or None
                group_row = await conn.fetchrow("""
                    SELECT group_data
                    FROM NPCGroups
                    WHERE group_id=$1 AND user_id=$2 AND conversation_id=$3
                """, group_id, self.user_id, self.conversation_id)

                if not group_row:
                    logger.warning(f"Group {group_id} not found for user {self.user_id}, convo {self.conversation_id}.")
                    return False

                group_data = group_row['group_data']
                if isinstance(group_data, str):
                    try:
                        group_data = json.loads(group_data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in group_data for group {group_id}.", exc_info=True)
                        return False
                elif group_data is None:
                    group_data = {}

                # Get NPC data
                npc_row = await conn.fetchrow("""
                    SELECT npc_name, dominance
                    FROM NPCStats
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                """, npc_id, self.user_id, self.conversation_id)

                if not npc_row:
                    logger.warning(f"NPC {npc_id} not found for user {self.user_id}, convo {self.conversation_id}.")
                    return False

                npc_name, dominance = npc_row['npc_name'], npc_row['dominance']

                # Add NPC to group data structure
                members = group_data.setdefault("members", [])

                # Check if NPC is already in group
                if any(member.get("npc_id") == npc_id for member in members):
                     logger.info(f"NPC {npc_id} already in group {group_id}. Skipping add.")
                     return True

                # Add new member
                members.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "joined_date": datetime.now().isoformat(),
                    "status": "active",
                    "role": role
                })

                # REFACTORED: Use LoreSystem to update group data
                lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
                result = await lore_system.propose_and_enact_change(
                    ctx=self.ctx,
                    entity_type="NPCGroups",
                    entity_identifier={"group_id": group_id, "user_id": self.user_id, "conversation_id": self.conversation_id},
                    updates={"group_data": json.dumps(group_data)},
                    reason=f"Added NPC {npc_name} to group as {role}"
                )

                if result.get("status") == "committed":
                    logger.info(f"Successfully added NPC {npc_id} to group {group_id}.")
                    return True
                else:
                    logger.error(f"Failed to add NPC {npc_id} to group {group_id}: {result}")
                    return False

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error adding NPC {npc_id} to group {group_id}: {db_err}", exc_info=True)
            return False
        except Exception as e:
            logger.exception(f"Unexpected error adding NPC {npc_id} to group {group_id}: {e}")
            return False
    
    async def generate_group_dynamics(self, group_id: int) -> Dict[str, Any]:
        """
        Asynchronously generate group dynamics for a group of NPCs.

        Args:
            group_id: ID of the group

        Returns:
            Dictionary with group dynamics or error info.
        """
        try:
            async with get_db_connection_context() as conn:
                # Get group data
                group_row = await conn.fetchrow("""
                    SELECT group_data, group_name
                    FROM NPCGroups
                    WHERE group_id=$1 AND user_id=$2 AND conversation_id=$3
                """, group_id, self.user_id, self.conversation_id)

                if not group_row:
                    logger.warning(f"Group {group_id} not found during dynamics generation.")
                    return {"error": "Group not found"}

                group_data = group_row['group_data']
                group_name = group_row['group_name']

                # Handle JSON loading
                if isinstance(group_data, str):
                    try:
                        group_data = json.loads(group_data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in group_data for group {group_id} during dynamics.", exc_info=True)
                        return {"error": "Invalid group data"}
                elif group_data is None:
                    group_data = {}

                # Generate dynamic relationships between members
                members = group_data.setdefault("members", [])
                member_ids = [m.get("npc_id") for m in members if m.get("npc_id") is not None]
            
                relationships = []
                # Use internal methods which already handle async DB access
                for i in range(len(member_ids)):
                    for j in range(i+1, len(member_ids)):
                        npc1_id = member_ids[i]
                        npc2_id = member_ids[j]

                        # Check existing relationship
                        rel = await self.get_relationship("npc", npc1_id, "npc", npc2_id)
                        if not rel:
                            # Create relationship
                            rel_type = random.choice(["neutral", "alliance", "rivalry", "dominant", "submission"])
                            rel = await self.create_relationship("npc", npc1_id, "npc", npc2_id, rel_type)

                        relationships.append(rel)

                # Generate random group events
                events = []
                if members:
                    for _ in range(3):
                        event_type = random.choice(["meeting", "conflict", "collaboration", "celebration", "crisis"])
                                
                        if event_type == "meeting":
                            events.append(f"The group held a meeting to discuss their goals and plans.")
                        elif event_type == "conflict":
                            conflict_members = random.sample(members, min(2, len(members)))
                            events.append(f"{conflict_members[0].get('npc_name')} and {conflict_members[1].get('npc_name')} had a disagreement about the group's direction.")
                        elif event_type == "collaboration":
                            events.append(f"The group worked together on a project, strengthening their bonds.")
                        elif event_type == "celebration":
                            events.append(f"The group celebrated a significant achievement together.")
                        elif event_type == "crisis":
                            events.append(f"The group faced a crisis that tested their unity and resolve.")
                
                # Update group dynamics
                dynamics = group_data.get("dynamics", {})
                if not dynamics:
                    dynamics = {
                        "hierarchy": random.randint(30, 70),
                        "cohesion": random.randint(30, 70),
                        "secrecy": random.randint(30, 70),
                        "territoriality": random.randint(30, 70),
                        "exclusivity": random.randint(30, 70)
                    }
                
                # Update group data
                group_data["dynamics"] = dynamics
                group_data["shared_history"] = group_data.get("shared_history", []) + events
                
                # REFACTORED: Use LoreSystem to update group data
                lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
                result = await lore_system.propose_and_enact_change(
                    ctx=self.ctx,
                    entity_type="NPCGroups",
                    entity_identifier={"group_id": group_id, "user_id": self.user_id, "conversation_id": self.conversation_id},
                    updates={"group_data": json.dumps(group_data)},
                    reason=f"Generated dynamics and events for group {group_name}"
                )
                
                if result.get("status") == "committed":
                    logger.info(f"Generated dynamics for group {group_id} ('{group_name}').")
                    return {
                        "group_id": group_id,
                        "group_name": group_name,
                        "dynamics": dynamics,
                        "events": events,
                        "relationships": relationships
                    }
                else:
                    logger.error(f"Failed to update group dynamics: {result}")
                    return {"error": "Failed to update group dynamics"}

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error generating dynamics for group {group_id}: {db_err}", exc_info=True)
            return {"error": f"Database error: {db_err}"}
        except Exception as e:
            logger.exception(f"Unexpected error generating dynamics for group {group_id}: {e}")
            return {"error": f"Unexpected error: {e}"}

        
    async def generate_relationship_evolution(self, link_id: int) -> Dict[str, Any]:
        """
        Asynchronously generate relationship evolution information.

        Args:
            link_id: ID of the relationship link

        Returns:
            Dictionary with evolution information or error info.
        """
        try:
            async with get_db_connection_context() as conn:
                # Get relationship data
                row = await conn.fetchrow("""
                    SELECT entity1_type, entity1_id, entity2_type, entity2_id,
                           link_type, link_level, link_history, dynamics
                    FROM SocialLinks
                    WHERE link_id=$1 AND user_id=$2 AND conversation_id=$3
                """, link_id, self.user_id, self.conversation_id)

                if not row:
                    logger.warning(f"Relationship link {link_id} not found for evolution.")
                    return {"error": "Relationship not found"}

                e1_type, e1_id, e2_type, e2_id = row['entity1_type'], row['entity1_id'], row['entity2_type'], row['entity2_id']
                link_type, link_level = row['link_type'], row['link_level']
                history_json, dynamics_json = row['link_history'], row['dynamics']
            
                # Get entity names
                e1_name_task = asyncio.create_task(self.get_entity_name(e1_type, e1_id))
                e2_name_task = asyncio.create_task(self.get_entity_name(e2_type, e2_id))
                e1_name, e2_name = await e1_name_task, await e2_name_task

                # Parse dynamics
                dynamics = {}
                if dynamics_json:
                    if isinstance(dynamics_json, str):
                         try: dynamics = json.loads(dynamics_json)
                         except json.JSONDecodeError: pass
                    else:
                         dynamics = dynamics_json

                # Parse history
                history_events = []
                if history_json:
                    if isinstance(history_json, str):
                         try: history_events = json.loads(history_json)
                         except json.JSONDecodeError: pass
                    else:
                         history_events = history_json

            
            # Generate potential future trajectories
            trajectories = []
            
            if "control" in dynamics and dynamics["control"] > 60:
                trajectories.append({
                    "name": "Increasing Control",
                    "description": f"{e1_name} gains even more control over {e2_name}",
                    "probability": "High",
                    "triggers": ["Extended isolation", "Emotional vulnerability", "Dependency reinforcement"]
                })
            
            if "trust" in dynamics and dynamics["trust"] > 70:
                trajectories.append({
                    "name": "Deep Trust",
                    "description": f"{e1_name} and {e2_name} develop profound trust",
                    "probability": "Medium",
                    "triggers": ["Shared vulnerability", "Consistent support", "Mutual secrets"]
                })
            
            if "tension" in dynamics and dynamics["tension"] > 50:
                trajectories.append({
                    "name": "Breaking Point",
                    "description": f"Tension between {e1_name} and {e2_name} reaches a critical threshold",
                    "probability": "Medium-High",
                    "triggers": ["Public confrontation", "Boundary violation", "Resource competition"]
                })
            
            # Generate default trajectory if none exist
            if not trajectories:
                trajectories.append({
                    "name": "Status Quo",
                    "description": f"Relationship between {e1_name} and {e2_name} continues as is",
                    "probability": "High",
                    "triggers": ["Routine maintenance", "Absence of disruption"]
                })
            
            # Generate relationship insights
            insights = []
            
            # Power dynamic insight
            if "control" in dynamics and "dependency" in dynamics:
                control = dynamics["control"]
                dependency = dynamics["dependency"]
                
                if control > 70 and dependency > 60:
                    insights.append(f"Strong power imbalance with {e1_name} controlling and {e2_name} dependent")
                elif control > 50 and dependency > 40:
                    insights.append(f"Moderate power dynamic with {e1_name} guiding and {e2_name} following")
                else:
                    insights.append(f"Balanced power dynamic between {e1_name} and {e2_name}")
            
            # Emotional dynamic insight
            if "intimacy" in dynamics and "trust" in dynamics:
                intimacy = dynamics["intimacy"]
                trust = dynamics["trust"]
                
                if intimacy > 70 and trust > 70:
                    insights.append(f"Deep emotional connection between {e1_name} and {e2_name}")
                elif intimacy > 50 and trust < 30:
                    insights.append(f"Intimate but untrusting relationship between {e1_name} and {e2_name}")
                elif intimacy < 30 and trust > 70:
                    insights.append(f"Trusted but distant relationship between {e1_name} and {e2_name}")
                else:
                    insights.append(f"Developing emotional dynamic between {e1_name} and {e2_name}")
            
            # Default insight if none exist
            if not insights:
                insights.append(f"Evolving relationship between {e1_name} and {e2_name}")
            
            return {
                "link_id": link_id,
                "entity1_name": e1_name,
                "entity2_name": e2_name,
                "relationship_type": link_type,
                "relationship_level": link_level,
                "dynamics": dynamics,
                "history_length": len(history_events),
                "recent_history": history_events[-3:] if len(history_events) >= 3 else history_events,
                "trajectories": trajectories,
                "insights": insights
            }
        except Exception as e:
            logging.error(f"Error generating relationship evolution: {e}")
            return {"error": str(e)}
    
    async def get_entity_name(self, entity_type: str, entity_id: int) -> str:
        """
        Asynchronously get the name of an entity.

        Args:
            entity_type: Type of entity
            entity_id: ID of entity

        Returns:
            Name of the entity or a default string.
        """
        # Handle player name directly
        if entity_type == "player":
            return "Player"

        # Fetch NPC name from DB
        try:
            async with get_db_connection_context() as conn:
                npc_name = await conn.fetchval("""
                    SELECT npc_name
                    FROM NPCStats
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                """, entity_id, self.user_id, self.conversation_id)

                return npc_name if npc_name else f"Unknown NPC ({entity_id})"

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error fetching name for {entity_type} {entity_id}: {db_err}", exc_info=True)
            return f"DB Error ({entity_type} {entity_id})"
        except Exception as e:
            logger.exception(f"Unexpected error fetching name for {entity_type} {entity_id}: {e}")
            return f"Error ({entity_type} {entity_id})"

    async def get_player_relationships(self) -> List[Dict[str, Any]]:
        """
        Asynchronously get all relationships involving the player.

        Returns:
            List of relationship data. Returns empty list on error.
        """
        relationships = []
        try:
            player_id = self.user_id

            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                           link_type, link_level
                    FROM SocialLinks
                    WHERE user_id=$1 AND conversation_id=$2
                    AND ((entity1_type='player' AND entity1_id=$3) OR
                         (entity2_type='player' AND entity2_id=$3))
                """, self.user_id, self.conversation_id, player_id)

                # Process rows concurrently
                tasks = []
                for row in rows:
                    link_id = row['link_id']
                    link_type = row['link_type']
                    link_level = row['link_level']

                    # Determine the NPC involved
                    if row['entity1_type'] == "player":
                        npc_type, npc_id = row['entity2_type'], row['entity2_id']
                    else:
                        npc_type, npc_id = row['entity1_type'], row['entity1_id']

                    # Create a task to get NPC name and append data
                    async def fetch_and_format(lt, ll, nt, ni, lid):
                         npc_name = await self.get_entity_name(nt, ni)
                         return {
                             "link_id": lid,
                             "npc_id": ni,
                             "npc_name": npc_name,
                             "relationship_type": lt,
                             "relationship_level": ll
                         }
                    tasks.append(fetch_and_format(link_type, link_level, npc_type, npc_id, link_id))

                # Gather results from all tasks
                if tasks:
                    relationships = await asyncio.gather(*tasks)

            return relationships

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error fetching player relationships for user {self.user_id}: {db_err}", exc_info=True)
            return []
        except Exception as e:
            logger.exception(f"Unexpected error fetching player relationships for user {self.user_id}: {e}")
            return []
