"""
Core narrative progression system for the Lore System.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..config.settings import config
from ..utils.db import execute_query, DatabaseError
from ..utils.cache import get_cached_value, set_cached_value, invalidate_cache_pattern

logger = logging.getLogger(__name__)

@dataclass
class NarrativeStage:
    """Represents a stage in the narrative progression."""
    name: str
    description: str
    required_corruption: int
    required_dependency: int
    events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.events = self.events or []

class NarrativeError(Exception):
    """Custom exception for narrative-related errors."""
    pass

class NarrativeProgression:
    """Manages narrative progression and stage transitions."""
    
    def __init__(self):
        """Initialize narrative stages from config."""
        self.stages = [
            NarrativeStage(**stage) for stage in config.NARRATIVE_STAGES
        ]
    
    async def get_current_stage(self, user_id: int, conversation_id: int) -> NarrativeStage:
        """
        Get the current narrative stage for a user.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Current narrative stage
            
        Raises:
            NarrativeError: If stage determination fails
        """
        cache_key = config.CACHE_KEYS["narrative_stage"].format(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Try to get from cache first
        cached_stage = await get_cached_value(cache_key)
        if cached_stage:
            return NarrativeStage(**cached_stage)
        
        try:
            # Get player stats from database
            query = """
                SELECT corruption, dependency
                FROM PlayerStats
                WHERE user_id = %(user_id)s 
                AND conversation_id = %(conversation_id)s
                AND player_name = 'Chase'
            """
            result = await execute_query(query, {
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            if not result:
                # No stats found, return first stage
                current_stage = self.stages[0]
            else:
                corruption, dependency = result[0]
                # Find the highest stage the player qualifies for
                current_stage = self.stages[0]
                for stage in self.stages:
                    if corruption >= stage.required_corruption and dependency >= stage.required_dependency:
                        current_stage = stage
                    else:
                        break
            
            # Cache the result
            await set_cached_value(cache_key, current_stage.__dict__)
            return current_stage
            
        except DatabaseError as e:
            logger.error(f"Failed to get current stage: {e}")
            raise NarrativeError(f"Failed to determine narrative stage: {str(e)}")
    
    async def check_for_stage_transition(self, user_id: int, conversation_id: int) -> Optional[NarrativeStage]:
        """
        Check if the player should transition to a new stage.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            New stage if transition should occur, None otherwise
            
        Raises:
            NarrativeError: If transition check fails
        """
        current_stage = await self.get_current_stage(user_id, conversation_id)
        
        try:
            # Get current stats
            query = """
                SELECT corruption, dependency
                FROM PlayerStats
                WHERE user_id = %(user_id)s 
                AND conversation_id = %(conversation_id)s
                AND player_name = 'Chase'
            """
            result = await execute_query(query, {
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            if not result:
                return None
                
            corruption, dependency = result[0]
            
            # Find the next stage if any
            current_index = self.stages.index(current_stage)
            if current_index < len(self.stages) - 1:
                next_stage = self.stages[current_index + 1]
                if (corruption >= next_stage.required_corruption and 
                    dependency >= next_stage.required_dependency):
                    return next_stage
            
            return None
            
        except DatabaseError as e:
            logger.error(f"Failed to check stage transition: {e}")
            raise NarrativeError(f"Failed to check stage transition: {str(e)}")
    
    async def apply_stage_transition(self, user_id: int, conversation_id: int, new_stage: NarrativeStage) -> None:
        """
        Apply a stage transition and record it.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            new_stage: New narrative stage
            
        Raises:
            NarrativeError: If transition application fails
        """
        try:
            # Record the transition
            query = """
                INSERT INTO NarrativeTransitions 
                (user_id, conversation_id, old_stage, new_stage, transition_time)
                VALUES (%(user_id)s, %(conversation_id)s, %(old_stage)s, %(new_stage)s, %(transition_time)s)
            """
            await execute_query(query, {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "old_stage": (await self.get_current_stage(user_id, conversation_id)).name,
                "new_stage": new_stage.name,
                "transition_time": datetime.utcnow()
            })
            
            # Invalidate cache
            cache_key = config.CACHE_KEYS["narrative_stage"].format(
                user_id=user_id,
                conversation_id=conversation_id
            )
            await invalidate_cache_pattern(cache_key)
            
        except DatabaseError as e:
            logger.error(f"Failed to apply stage transition: {e}")
            raise NarrativeError(f"Failed to apply stage transition: {str(e)}")
    
    async def get_stage_events(self, stage: NarrativeStage) -> List[Dict[str, Any]]:
        """
        Get events associated with a narrative stage.
        
        Args:
            stage: Narrative stage
            
        Returns:
            List of stage events
            
        Raises:
            NarrativeError: If event retrieval fails
        """
        try:
            query = """
                SELECT event_id, event_type, description, requirements
                FROM StageEvents
                WHERE stage_name = %(stage_name)s
                ORDER BY event_id
            """
            results = await execute_query(query, {"stage_name": stage.name})
            
            return [
                {
                    "event_id": row[0],
                    "event_type": row[1],
                    "description": row[2],
                    "requirements": row[3]
                }
                for row in results
            ]
            
        except DatabaseError as e:
            logger.error(f"Failed to get stage events: {e}")
            raise NarrativeError(f"Failed to get stage events: {str(e)}")

# Create global narrative progression instance
narrative_progression = NarrativeProgression()
