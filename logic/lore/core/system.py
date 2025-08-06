# logic/lore/core/system.py

"""
Main Lore System class that integrates all components.
Updated to use the new dynamic relationship system.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..config.settings import config
from .narrative import narrative_progression, NarrativeStage, NarrativeError
from ..utils.cache import invalidate_cache_pattern

logger = logging.getLogger(__name__)

class LoreError(Exception):
    """Custom exception for lore system errors."""
    pass

class LoreSystem:
    """Main class that integrates all lore system components."""
    
    def __init__(self):
        """Initialize the lore system."""
        self.narrative = narrative_progression
        # Remove social_links_manager dependency - we'll use OptimizedRelationshipManager directly
        self._relationship_manager = None
    
    async def _get_relationship_manager(self, user_id: int, conversation_id: int):
        """Lazy load relationship manager."""
        if self._relationship_manager is None or \
           self._relationship_manager.user_id != user_id or \
           self._relationship_manager.conversation_id != conversation_id:
            from logic.dynamic_relationships import OptimizedRelationshipManager
            self._relationship_manager = OptimizedRelationshipManager(user_id, conversation_id)
        return self._relationship_manager
    
    async def get_current_state(
        self,
        user_id: int,
        conversation_id: int
    ) -> Dict[str, Any]:
        """
        Get the current state of the lore system for a user.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Dictionary containing current narrative stage and relationships
            
        Raises:
            LoreError: If state retrieval fails
        """
        try:
            # Get current narrative stage
            current_stage = await self.narrative.get_current_stage(user_id, conversation_id)
            
            # Get relationship manager
            manager = await self._get_relationship_manager(user_id, conversation_id)
            
            # Get all relationships for the player using the new system
            from db.connection import get_db_connection_context
            
            relationships = []
            async with get_db_connection_context() as conn:
                # Find all relationships involving the player
                rows = await conn.fetch("""
                    SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                           dynamics, patterns, archetypes, last_interaction
                    FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                    AND (
                        (entity1_type = 'player' AND entity1_id = 1)
                        OR (entity2_type = 'player' AND entity2_id = 1)
                    )
                """, user_id, conversation_id)
                
                for row in rows:
                    # Determine the other entity
                    if row['entity1_type'] == 'player':
                        entity_type = row['entity2_type']
                        entity_id = row['entity2_id']
                    else:
                        entity_type = row['entity1_type']
                        entity_id = row['entity1_id']
                    
                    # Get the full state
                    state = await manager.get_relationship_state(
                        entity1_type='player',
                        entity1_id=1,
                        entity2_type=entity_type,
                        entity2_id=entity_id
                    )
                    
                    relationships.append({
                        "link_id": state.link_id,
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "dimensions": state.dimensions.to_dict(),
                        "patterns": list(state.history.active_patterns),
                        "archetypes": list(state.active_archetypes),
                        "momentum": state.momentum.get_magnitude(),
                        "last_interaction": state.last_interaction.isoformat()
                    })
            
            # Get stage events
            stage_events = await self.narrative.get_stage_events(current_stage)
            
            return {
                "narrative_stage": {
                    "name": current_stage.name,
                    "description": current_stage.description,
                    "required_corruption": current_stage.required_corruption,
                    "required_dependency": current_stage.required_dependency,
                    "events": stage_events
                },
                "relationships": relationships
            }
            
        except (NarrativeError, Exception) as e:
            logger.error(f"Failed to get current state: {e}")
            raise LoreError(f"Failed to retrieve current state: {str(e)}")
    
    async def update_relationship(
        self,
        user_id: int,
        conversation_id: int,
        entity_type: str,
        entity_id: int,
        interaction: Dict[str, Any] = None,
        dimension_changes: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Update a relationship using the new dynamic system."""
        manager = await self._get_relationship_manager(user_id, conversation_id)
        
        if interaction:
            # Process as interaction
            result = await manager.process_interaction(
                entity1_type="player",
                entity1_id=1,
                entity2_type=entity_type,
                entity2_id=entity_id,
                interaction=interaction
            )
        elif dimension_changes:
            # Direct dimension update
            state = await manager.get_relationship_state(
                entity1_type="player",
                entity1_id=1,
                entity2_type=entity_type,
                entity2_id=entity_id
            )
            
            for dim, change in dimension_changes.items():
                if hasattr(state.dimensions, dim):
                    current = getattr(state.dimensions, dim)
                    setattr(state.dimensions, dim, current + change)
            
            state.dimensions.clamp()
            await manager._queue_update(state)
            await manager._flush_updates()
            
            result = {"success": True, "changes": dimension_changes}
        else:
            result = {"success": False, "error": "No interaction or dimension changes provided"}
        
        return result
    
    async def get_available_events(
        self,
        user_id: int,
        conversation_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get all available events for the current narrative stage.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            List of available events
            
        Raises:
            LoreError: If event retrieval fails
        """
        try:
            current_stage = await self.narrative.get_current_stage(
                user_id, conversation_id
            )
            return await self.narrative.get_stage_events(current_stage)
            
        except NarrativeError as e:
            logger.error(f"Failed to get available events: {e}")
            raise LoreError(f"Failed to retrieve available events: {str(e)}")
    
    async def get_relationship_network(
        self,
        user_id: int,
        conversation_id: int,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Get the relationship network for an entity using the new system.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            entity_type: Type of entity
            entity_id: ID of entity
            
        Returns:
            Dictionary containing relationship network
            
        Raises:
            LoreError: If network retrieval fails
        """
        try:
            manager = await self._get_relationship_manager(user_id, conversation_id)
            
            from db.connection import get_db_connection_context
            
            # Build network
            network = {
                "entity": {
                    "type": entity_type,
                    "id": entity_id
                },
                "relationships": []
            }
            
            async with get_db_connection_context() as conn:
                # Find all relationships for this entity
                rows = await conn.fetch("""
                    SELECT entity1_type, entity1_id, entity2_type, entity2_id
                    FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                    AND (
                        (entity1_type = $3 AND entity1_id = $4)
                        OR (entity2_type = $3 AND entity2_id = $4)
                    )
                """, user_id, conversation_id, entity_type, entity_id)
                
                for row in rows:
                    # Determine the other entity
                    if row['entity1_type'] == entity_type and row['entity1_id'] == entity_id:
                        other_type = row['entity2_type']
                        other_id = row['entity2_id']
                    else:
                        other_type = row['entity1_type']
                        other_id = row['entity1_id']
                    
                    # Get the relationship state
                    state = await manager.get_relationship_state(
                        entity1_type=entity_type,
                        entity1_id=entity_id,
                        entity2_type=other_type,
                        entity2_id=other_id
                    )
                    
                    network["relationships"].append({
                        "entity": {
                            "type": other_type,
                            "id": other_id
                        },
                        "dimensions": state.dimensions.to_dict(),
                        "patterns": list(state.history.active_patterns),
                        "archetypes": list(state.active_archetypes),
                        "momentum": state.momentum.get_magnitude(),
                        "last_interaction": state.last_interaction.isoformat()
                    })
            
            return network
            
        except Exception as e:
            logger.error(f"Failed to get relationship network: {e}")
            raise LoreError(f"Failed to retrieve relationship network: {str(e)}")
    
    async def process_lore_event(
        self,
        user_id: int,
        conversation_id: int,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a lore-related event.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            event_data: Event data
            
        Returns:
            Processing result
        """
        try:
            event_type = event_data.get("type", "unknown")
            
            if event_type == "relationship_change":
                # Handle relationship changes
                return await self.update_relationship(
                    user_id,
                    conversation_id,
                    event_data.get("entity_type"),
                    event_data.get("entity_id"),
                    interaction=event_data.get("interaction"),
                    dimension_changes=event_data.get("dimension_changes")
                )
            elif event_type == "narrative_progression":
                # Check for stage transition
                new_stage = await self.narrative.check_for_stage_transition(
                    user_id, conversation_id
                )
                if new_stage:
                    await self.narrative.apply_stage_transition(
                        user_id, conversation_id, new_stage
                    )
                    return {
                        "success": True,
                        "transition": True,
                        "new_stage": new_stage.name
                    }
                return {
                    "success": True,
                    "transition": False
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown event type: {event_type}"
                }
                
        except Exception as e:
            logger.error(f"Failed to process lore event: {e}")
            return {"success": False, "error": str(e)}
    
    async def advance_time(
        self,
        user_id: int,
        conversation_id: int,
        time_amount: int,
        time_unit: str = "hours"
    ) -> Dict[str, Any]:
        """
        Advance time and apply relationship drift.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            time_amount: Amount of time to advance
            time_unit: Unit of time (hours, days)
            
        Returns:
            Result of time advancement
        """
        try:
            manager = await self._get_relationship_manager(user_id, conversation_id)
            
            # Convert to days if needed
            if time_unit == "hours":
                days = time_amount / 24
            elif time_unit == "days":
                days = time_amount
            else:
                days = 1
            
            # Apply drift if a day or more has passed
            if days >= 1:
                await manager.apply_daily_drift()
                await manager._flush_updates()
            
            return {
                "success": True,
                "time_advanced": f"{time_amount} {time_unit}",
                "drift_applied": days >= 1
            }
            
        except Exception as e:
            logger.error(f"Failed to advance time: {e}")
            return {"success": False, "error": str(e)}

# Create global lore system instance
lore_system = LoreSystem()
