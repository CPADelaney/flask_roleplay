# lore/core/lore_system.py
"""
The unified LoreSystem Orchestrator. This is the primary, public-facing entry point
for all lore-related operations. It ensures consistency by using the Canon for
writes and the Registry for specialized generation tasks. Its core is the
generic `propose_and_enact_change` method.
"""
import logging
import json
from typing import Dict, Any, Optional, List

from agents import RunContextWrapper
from lore.core.registry import ManagerRegistry
from lore.core import canon
from db.connection import get_db_connection_context
from nyx.nyx_governance import AgentType
from nyx.governance_helpers import with_governance

logger = logging.getLogger(__name__)

class LoreSystem:
    _instances: Dict[str, "LoreSystem"] = {}

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.registry = ManagerRegistry(user_id, conversation_id)
        self.initialized = False

    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> "LoreSystem":
        key = f"{user_id}:{conversation_id}"
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            await instance.ensure_initialized()
            cls._instances[key] = instance
        return cls._instances[key]

    async def ensure_initialized(self):
        if self.initialized:
            return
        self.initialized = True
        logger.info(f"LoreSystem initialized for user {self.user_id}, conversation {self.conversation_id}.")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="propose_and_enact_change",
        action_description="Proposing change to {entity_type} based on: {reason}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def propose_and_enact_change(
        self,
        ctx,
        entity_type: str,
        entity_identifier: Dict[str, Any],
        updates: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """
        The universal method for making a canonical change to the world state.
        """
        # Ensure we have a proper context object
        if isinstance(ctx, str) or not hasattr(ctx, 'context'):
            # Create a proper context object if we received something invalid
            ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            logger.warning(f"Invalid context passed to propose_and_enact_change, created fallback context")
        
        logger.info(f"Proposing change to {entity_type} ({entity_identifier}) because: {reason}")
    
        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    # Step 1: Find the current state of the entity
                    where_clauses = [f"{key} = ${i+1}" for i, key in enumerate(entity_identifier.keys())]
                    where_sql = " AND ".join(where_clauses)
                    select_query = f"SELECT * FROM {entity_type} WHERE {where_sql}"
                    
                    existing_entity = await conn.fetchrow(select_query, *entity_identifier.values())
    
                    if not existing_entity:
                        return {"status": "error", "message": f"Entity not found: {entity_type} with {entity_identifier}"}
    
                    # Step 2: Validate for conflicts
                    conflicts = []
                    for field, new_value in updates.items():
                        if field in existing_entity and existing_entity[field] is not None and existing_entity[field] != new_value:
                            conflict_detail = (
                                f"Conflict on field '{field}'. "
                                f"Current value: '{existing_entity[field]}'. "
                                f"Proposed value: '{new_value}'."
                            )
                            conflicts.append(conflict_detail)
                            logger.warning(f"Conflict detected for {entity_type} ({entity_identifier}): {conflict_detail}")
    
                    if conflicts:
                        # Step 3a: Handle conflict by generating a new narrative event
                        dynamics = await self.registry.get_lore_dynamics()
                        conflict_description = f"A conflict arose: {reason}. Details: {', '.join(conflicts)}"
                        await dynamics.evolve_lore_with_event.fn(ctx, conflict_description)
                        return {"status": "conflict_generated", "details": conflicts}
    
                    # Step 3b: No conflict, commit the change
                    # Add type conversion for known boolean columns
                    boolean_columns = ['introduced', 'is_active', 'is_consolidated', 'is_archived']
                    
                    # Build SET clauses with proper casting for boolean columns
                    set_clauses = []
                    update_values = []
                    
                    for i, (key, value) in enumerate(updates.items()):
                        if key in boolean_columns:
                            # Ensure it's a boolean and use explicit cast
                            bool_value = bool(value) if not isinstance(value, bool) else value
                            set_clauses.append(f"{key} = ${i+1}::boolean")
                            update_values.append(bool_value)
                        else:
                            set_clauses.append(f"{key} = ${i+1}")
                            update_values.append(value)
                    
                    set_sql = ", ".join(set_clauses)
                    
                    # Build WHERE clause with proper placeholder numbering
                    # Start numbering after the SET clause placeholders
                    where_clauses_update = []
                    num_updates = len(updates)
                    for i, key in enumerate(entity_identifier.keys()):
                        placeholder_num = num_updates + i + 1
                        where_clauses_update.append(f"{key} = ${placeholder_num}")
                    where_sql_update = " AND ".join(where_clauses_update)
                    
                    # Add the entity identifier values
                    update_values.extend(list(entity_identifier.values()))
                    
                    update_query = f"UPDATE {entity_type} SET {set_sql} WHERE {where_sql_update}"
                    await conn.execute(update_query, *update_values)
    
                    # Step 4: Log the change as a canonical event in unified memory
                    event_text = f"The {entity_type} identified by {entity_identifier} was updated. Reason: {reason}. Changes: {updates}"
                    await canon.log_canonical_event(ctx, conn, event_text, tags=[entity_type.lower(), 'state_change'], significance=8)
            
            # Step 5: Propagate consequences to other systems (outside the transaction)
            # Create a more specific event description that will pass validation
            event_description = self._create_detailed_event_description(
                entity_type, entity_identifier, updates, reason
            )
            
            # Only propagate if the event is significant enough
            if event_description:
                dynamics = await self.registry.get_lore_dynamics()
                await dynamics.evolve_lore_with_event.fn(ctx, event_description)
            
            return {"status": "committed", "entity_type": entity_type, "identifier": entity_identifier, "changes": updates}
    
        except Exception as e:
            logger.exception(f"Failed to enact change for {entity_type} ({entity_identifier}): {e}")
            return {"status": "error", "message": str(e)}

    def _create_detailed_event_description(
        self, 
        entity_type: str, 
        entity_identifier: Dict[str, Any], 
        updates: Dict[str, Any], 
        reason: str
    ) -> Optional[str]:
        """
        Create a detailed event description that will pass validation.
        Returns None if the change is too minor to warrant lore evolution.
        """
        # Determine if this change is significant enough
        significant_fields = {
            'NPCStats': ['power_level', 'status', 'loyalty'],
            'Factions': ['leader_npc_id', 'power_level', 'territory', 'allies', 'rivals'],
            'Nations': ['leader_npc_id', 'government_type', 'stability'],
            'Locations': ['controlling_faction', 'strategic_value'],
            'NPCs': ['status', 'faction_affiliation', 'current_location']
        }
        
        # Check if any significant fields were updated
        if entity_type in significant_fields:
            updated_significant_fields = [
                field for field in significant_fields[entity_type] 
                if field in updates
            ]
            if not updated_significant_fields:
                # Not significant enough for lore evolution
                return None
        
        # Build a detailed event description based on the entity type and changes
        event_parts = []
        
        if entity_type == "NPCStats" and "power_level" in updates:
            event_parts.append(f"A significant shift in power has occurred - someone's influence has changed dramatically")
        elif entity_type == "Factions" and "leader_npc_id" in updates:
            event_parts.append(f"Leadership has changed within a major faction")
        elif entity_type == "Nations" and "leader_npc_id" in updates:
            event_parts.append(f"A nation has experienced a change in leadership")
        elif entity_type == "Locations" and "controlling_faction" in updates:
            event_parts.append(f"Control of a strategic location has shifted to new hands")
        else:
            # Generic but still specific
            event_parts.append(f"The {entity_type.lower()} structure has undergone important changes")
        
        # Add the reason
        event_parts.append(f"This occurred because: {reason}")
        
        # Add specific change details for context
        change_details = []
        for field, value in updates.items():
            if field in ['power_level', 'status', 'loyalty', 'stability']:
                change_details.append(f"{field.replace('_', ' ')} has shifted")
            elif field in ['leader_npc_id', 'controlling_faction']:
                change_details.append(f"new leadership has been established")
            elif field in ['territory', 'allies', 'rivals']:
                change_details.append(f"{field} relationships have changed")
        
        if change_details:
            event_parts.append(f"Specifically: {', '.join(change_details)}")
        
        # Add context about the broader implications
        if entity_type in ["Nations", "Factions"]:
            event_parts.append("This shift in power will ripple through the political landscape, affecting alliances and rivalries")
        elif entity_type == "Locations":
            event_parts.append("This territorial change may spark new conflicts or opportunities in the region")
        elif entity_type == "NPCStats":
            event_parts.append("This personal transformation will influence their relationships and standing in society")
        
        return ". ".join(event_parts)

    # --- Convenience Wrappers (Optional but Recommended) ---
    # These methods provide a clean, high-level API but all use the generic method internally.

    async def execute_coup(self, ctx, nation_id: int, new_leader_id: int, reason: str):
        """A high-level wrapper for a coup event."""
        # Ensure proper context
        if isinstance(ctx, str) or not hasattr(ctx, 'context'):
            ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
        
        return await self.propose_and_enact_change(
            ctx=ctx,
            entity_type="Nations",
            entity_identifier={"id": nation_id},
            updates={"leader_npc_id": new_leader_id},
            reason=reason
        )

    async def assign_faction_territory(self, ctx, faction_id: int, location_name: str, reason: str):
        """A high-level wrapper for a faction taking over territory."""
        # Ensure proper context
        if isinstance(ctx, str) or not hasattr(ctx, 'context'):
            ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
        
        location_id = None
        # First, ensure the location exists canonically
        async with get_db_connection_context() as conn:
            location_id = await canon.find_or_create_location(ctx, conn, location_name)

        return await self.propose_and_enact_change(
            ctx=ctx,
            entity_type="Factions",
            entity_identifier={"id": faction_id},
            updates={"territory": location_name}, # Assuming territory is a text field
            reason=reason
        )
