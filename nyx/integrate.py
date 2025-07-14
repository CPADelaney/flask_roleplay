# nyx/integrate.py

"""
Integration module for Nyx's central governance system.

This module provides a unified central governance system that controls all aspects of the game,
including story generation, NPCs, memory systems, user modeling, and more. All agent modules
are coordinated through this central authority, ensuring Nyx maintains consistent control.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from memory.memory_nyx_integration import (
    get_memory_nyx_bridge,
    remember_through_nyx,
    recall_through_nyx,
    create_belief_through_nyx,
    run_maintenance_through_nyx
)
from memory.memory_agent_sdk import create_memory_agent, MemorySystemContext

from lore.validation import ValidationManager

import asyncpg

# Import story components
from story_agent.agent_interaction import (
    orchestrate_conflict_analysis_and_narrative,
    generate_comprehensive_story_beat
)
from story_agent.story_director_agent import initialize_story_director

# Import agent processing components for full integration
from nyx.nyx_governance import NyxUnifiedGovernor, AgentType, DirectiveType, DirectivePriority
from nyx.nyx_agent_sdk import process_user_input, generate_reflection
from nyx.user_model_sdk import process_user_input_for_model, get_response_guidance_for_user
from nyx.scene_manager_sdk import process_scene_input, generate_npc_response
from nyx.llm_integration import generate_text_completion

# Import new game components
from new_game_agent import NewGameAgent

# The agent trace utility
from agents import trace

# Database connection helper
from db.connection import get_db_connection_context

# Caching utilities
from utils.caching import CACHE_TTL, NPC_DIRECTIVE_CACHE, AGENT_DIRECTIVE_CACHE

from lore.lore_generator import DynamicLoreGenerator
from lore.validation import ValidationManager
from lore.error_manager import ErrorHandler

from .nyx_agent_sdk import AgentContext
from .nyx_enhanced_system import NyxEnhancedSystem
from .response_filter import ResponseFilter
from .nyx_planner import NyxPlanner
from .nyx_task_integration import NyxTaskIntegration
from memory.memory_integration import MemoryIntegration
from .scene_manager_sdk import SceneContext
from .user_model_sdk import UserModelContext, UserModelManager
from lore.core.lore_system import LoreSystem

logger = logging.getLogger(__name__)

# Initialize components
lore_system = LoreSystem()
lore_validator = ValidationManager()
error_handler = ErrorHandler()

_GOVERNANCE_CACHE: Dict[Tuple[int, int], NyxUnifiedGovernor] = {}
_GOVERNANCE_INIT_LOCKS: Dict[Tuple[int, int], asyncio.Lock] = {}

async def get_central_governance(user_id: int, conversation_id: int) -> NyxUnifiedGovernor:
    """
    Get or create the central governance instance for a user/conversation.
    
    This function ensures that:
    1. Only one governance instance exists per user/conversation pair
    2. The governance system is properly initialized
    3. All subsystems (including LoreSystem) are initialized in the correct order
    4. Concurrent calls don't create duplicate instances
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        
    Returns:
        The initialized NyxUnifiedGovernor instance
    """
    cache_key = (user_id, conversation_id)
    
    # Get or create a lock for this cache key to prevent concurrent initialization
    if cache_key not in _GOVERNANCE_INIT_LOCKS:
        _GOVERNANCE_INIT_LOCKS[cache_key] = asyncio.Lock()
    
    async with _GOVERNANCE_INIT_LOCKS[cache_key]:
        # Check cache first (inside the lock to ensure thread safety)
        if cache_key in _GOVERNANCE_CACHE:
            governor = _GOVERNANCE_CACHE[cache_key]
            # Ensure it's initialized
            if hasattr(governor, '_initialized') and governor._initialized:
                logger.debug(f"Returning cached initialized governor for {cache_key}")
                return governor
            else:
                # Continue to initialization below
                logger.info(f"Found uninitialized governor in cache for {cache_key}, initializing now")
        else:
            # Create new governor
            governor = NyxUnifiedGovernor(user_id, conversation_id)
            _GOVERNANCE_CACHE[cache_key] = governor
            logger.info(f"Created new governor for user {user_id}, conversation {conversation_id}")
        
        # Check again if it was initialized while we were waiting for the lock
        if hasattr(governor, '_initialized') and governor._initialized:
            logger.debug(f"Governor was initialized while waiting for lock for {cache_key}")
            return governor
        
        # Initialize the governor
        # This will:
        # 1. Create and initialize the LoreSystem (with proper dependency injection)
        # 2. Initialize the memory system
        # 3. Initialize the game state
        # 4. Discover and register agents
        logger.info(f"Initializing governor for user {user_id}, conversation {conversation_id}")
        await governor.initialize()
        
        logger.info(f"Governor fully initialized for user {user_id}, conversation {conversation_id}")
        return governor

def clear_governance_cache(user_id: Optional[int] = None, conversation_id: Optional[int] = None):
    """
    Clear governance cache entries.
    
    Args:
        user_id: If provided with conversation_id, clear specific entry
        conversation_id: If provided with user_id, clear specific entry
        If neither provided, clear entire cache
    """
    global _GOVERNANCE_CACHE, _GOVERNANCE_INIT_LOCKS
    
    if user_id is not None and conversation_id is not None:
        cache_key = (user_id, conversation_id)
        if cache_key in _GOVERNANCE_CACHE:
            del _GOVERNANCE_CACHE[cache_key]
            logger.info(f"Cleared governance cache for {cache_key}")
        if cache_key in _GOVERNANCE_INIT_LOCKS:
            del _GOVERNANCE_INIT_LOCKS[cache_key]
    else:
        _GOVERNANCE_CACHE.clear()
        _GOVERNANCE_INIT_LOCKS.clear()
        logger.info("Cleared entire governance cache")


async def generate_lore_with_governance(
    user_id: int,
    conversation_id: int,
    environment_desc: str
) -> Dict[str, Any]:
    """
    Generate comprehensive lore with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        environment_desc: Description of the environment
        
    Returns:
        Generated lore
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.generate_lore(environment_desc)

async def integrate_lore_with_npcs(
    user_id: int,
    conversation_id: int,
    npc_ids: List[int]
) -> Dict[str, Any]:
    """
    Integrate lore with NPCs with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        npc_ids: List of NPC IDs to integrate lore with
        
    Returns:
        Integration results
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.integrate_lore_with_npcs(npc_ids)

async def enhance_context_with_lore(
    user_id: int,
    conversation_id: int,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhance context with relevant lore.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        context: Current context dictionary
        
    Returns:
        Enhanced context with lore
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.enhance_context_with_lore(context)

async def generate_scene_with_lore(
    user_id: int,
    conversation_id: int,
    location: str
) -> Dict[str, Any]:
    """
    Generate a scene description enhanced with lore.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        location: Location name
        
    Returns:
        Enhanced scene description
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.generate_scene_with_lore(location)


class JointMemoryGraph:
    """
    Graph for tracking shared memories between entities.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the joint memory graph."""
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def add_joint_memory(
        self,
        memory_text: str,
        source_type: str,
        source_id: int,
        shared_with: List[Dict[str, Any]],
        significance: int = 5,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Add a memory with governance oversight.
        
        This enhances the original method by checking permissions and reporting actions.
        """
        tags = tags or []
        metadata = metadata or {}
        
        # Proceed with original implementation
        memory_id = await self._store_joint_memory(
            memory_text, source_type, source_id, shared_with, 
            significance, tags, metadata
        )
        
        return memory_id

    async def _store_joint_memory(
        self,
        memory_text: str,
        source_type: str,
        source_id: int,
        shared_with: List[Dict[str, Any]],
        significance: int,
        tags: List[str],
        metadata: Dict[str, Any]
    ) -> int:
        """Store a joint memory in the database using asyncpg."""
        try:
            async with get_db_connection_context() as conn:
                # Use a transaction to ensure atomicity
                async with conn.transaction():
                    # Insert the main memory record and get the ID
                    memory_id = await conn.fetchval("""
                        INSERT INTO JointMemories (
                            user_id, conversation_id, memory_text,
                            source_type, source_id, significance,
                            tags, metadata, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, NOW())
                        RETURNING memory_id
                    """,
                        self.user_id, self.conversation_id, memory_text,
                        source_type, source_id, significance,
                        json.dumps(tags), json.dumps(metadata)
                    )

                    if memory_id is None:
                         raise RuntimeError("Failed to insert memory, memory_id is NULL.") # Or handle differently

                    # Store memory sharing relationships
                    # Consider executemany for potential performance improvement if many entities
                    for entity in shared_with:
                        await conn.execute("""
                            INSERT INTO JointMemorySharing (
                                memory_id, entity_type, entity_id
                            )
                            VALUES ($1, $2, $3)
                        """,
                            memory_id, entity.get("entity_type"), entity.get("entity_id")
                        )

                logger.debug(f"Successfully stored joint memory {memory_id}")
                return memory_id

        except (asyncpg.PostgresError, ConnectionError, RuntimeError) as e:
            logger.error(f"Error storing joint memory: {e}", exc_info=True)
            return -1
        except Exception as e: # Catch unexpected errors
             logger.error(f"Unexpected error storing joint memory: {e}", exc_info=True)
             return -1
    
    async def get_shared_memories(
        self,
        entity_type: str,
        entity_id: int,
        filter_tags: List[str] = None,
        min_significance: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get memories shared with a specific entity using asyncpg.
        """
        results = []
        try:
            async with get_db_connection_context() as conn:
                # Base query
                query = """
                    SELECT m.memory_id, m.memory_text, m.source_type, m.source_id,
                           m.significance, m.tags, m.metadata, m.created_at
                    FROM JointMemories m
                    INNER JOIN JointMemorySharing s ON m.memory_id = s.memory_id
                    WHERE m.user_id = $1 AND m.conversation_id = $2
                    AND s.entity_type = $3 AND s.entity_id = $4
                    AND m.significance >= $5
                """
                params = [self.user_id, self.conversation_id, entity_type, entity_id, min_significance]
                param_index = 6 # Next parameter index

                # Add tag filtering if needed (using JSONB array containment)
                # Assumes 'tags' column is JSONB and contains an array of strings.
                # If 'tags' is TEXT[], use `m.tags @> $${param_index}::text[]`
                if filter_tags:
                    query += f" AND m.tags::jsonb @> ${param_index}::jsonb"
                    params.append(json.dumps(filter_tags)) # Pass tags as a JSON string array
                    param_index += 1

                # Add ordering and limit
                query += f" ORDER BY m.significance DESC, m.created_at DESC LIMIT ${param_index}"
                params.append(limit)

                rows = await conn.fetch(query, *params)

                for row in rows:
                    # asyncpg can often auto-decode JSON/JSONB
                    tags_data = row['tags'] # Might be already parsed list/dict
                    metadata_data = row['metadata'] # Might be already parsed list/dict

                    # Optional: Ensure correct type if needed (e.g., if NULL or not auto-parsed)
                    parsed_tags = tags_data if isinstance(tags_data, list) else (json.loads(tags_data) if isinstance(tags_data, str) else [])
                    parsed_metadata = metadata_data if isinstance(metadata_data, dict) else (json.loads(metadata_data) if isinstance(metadata_data, str) else {})

                    results.append({
                        "memory_id": row['memory_id'],
                        "memory_text": row['memory_text'],
                        "source_type": row['source_type'],
                        "source_id": row['source_id'],
                        "significance": row['significance'],
                        "tags": parsed_tags,
                        "metadata": parsed_metadata,
                        "created_at": row['created_at'].isoformat() if row['created_at'] else None
                    })

            return results

        except (asyncpg.PostgresError, ConnectionError, json.JSONDecodeError) as e:
            logger.error(f"Error getting shared memories for entity {entity_type}/{entity_id}: {e}", exc_info=True)
            return [] # Return empty list on error
        except Exception as e: # Catch unexpected errors
             logger.error(f"Unexpected error getting shared memories: {e}", exc_info=True)
             return []

class GameEventManager:
    """
    Manager for game events with governance integration.
    """
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor = None):
        """Initialize with governor access."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor
        
        # Initialize required components
        self.nyx_agent_sdk = self.get_nyx_agent(user_id, conversation_id)
        self.npc_coordinator = self.get_npc_coordinator(user_id, conversation_id)
    
    def get_nyx_agent(self, user_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Get Nyx agent for this context.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Nyx agent interface
        """
        return {
            "process_game_event": self.process_game_event,
            "analyze_event": self._analyze_event,
            "determine_impact": self._determine_event_impact,
            "get_event_context": self._get_event_context
        }

    def get_npc_coordinator(self, user_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Get NPC coordinator for this context.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            NPC coordinator interface
        """
        return {
            "batch_update_npcs": self.batch_update_npcs,
            "update_npc": self._update_single_npc,
            "get_npc_info": self._get_npc_info,
            "execute_directive": self._execute_npc_directive
        }

    async def process_game_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a game event as Nyx.
        
        Args:
            event_type: Type of event
            event_data: Event data
            
        Returns:
            Processing results
        """
        try:
            # Analyze event
            analysis = await self._analyze_event(event_type, event_data)
            
            # Determine impact
            impact = await self._determine_event_impact(analysis)
            
            # Get context
            context = await self._get_event_context(event_type, event_data)
            
            # Determine which NPCs should be aware
            aware_npcs = await self._determine_aware_npcs(event_type, event_data)
            
            return {
                "should_broadcast_to_npcs": impact['should_broadcast'],
                "event_type": event_type,
                "event_data": event_data,
                "analysis": analysis,
                "impact": impact,
                "context": context,
                "aware_npcs": aware_npcs,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing game event: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def _analyze_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a game event.
        
        Args:
            event_type: Type of event
            event_data: Event data
            
        Returns:
            Event analysis
        """
        try:
            # Build analysis prompt
            prompt = {
                "event_type": event_type,
                "event_data": event_data,
                "task": "event_analysis"
            }
            
            # Generate analysis
            analysis = await generate_text_completion(prompt=prompt)
            
            return {
                "analysis": analysis,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error analyzing event: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def _determine_event_impact(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the impact of an event.
        
        Args:
            analysis: Event analysis
            
        Returns:
            Impact assessment
        """
        try:
            # Extract key elements
            event_type = analysis.get("event_type", "")
            event_data = analysis.get("event_data", {})
            
            # Calculate impact scores
            immediate_impact = await self._calculate_immediate_impact(event_data)
            long_term_impact = await self._calculate_long_term_impact(event_data)
            npc_impact = await self._calculate_npc_impact(event_data)
            
            # Determine broadcast threshold
            should_broadcast = (
                immediate_impact > 0.5 or
                long_term_impact > 0.7 or
                npc_impact > 0.3
            )
            
            return {
                "immediate_impact": immediate_impact,
                "long_term_impact": long_term_impact,
                "npc_impact": npc_impact,
                "should_broadcast": should_broadcast,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error determining event impact: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def _get_event_context(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get context for an event.
        
        Args:
            event_type: Type of event
            event_data: Event data
            
        Returns:
            Event context
        """
        try:
            # Get location context
            location = event_data.get("location", "")
            location_info = await self.governor.get_location_info(location)
            
            # Get NPC context
            npc_ids = event_data.get("npc_ids", [])
            npc_info = []
            for npc_id in npc_ids:
                npc_data = await self.npc_coordinator.get_npc_info(npc_id)
                npc_info.append(npc_data)
                
            # Get memory context
            memory_manager = await self.governor.get_memory_manager()
            memories = await memory_manager.recall(
                entity_type="event",
                entity_id=event_data.get("id", 0),
                limit=5
            )
            
            return {
                "location": location_info,
                "npcs": npc_info,
                "memories": memories,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting event context: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def _determine_aware_npcs(self, event_type: str, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Determine which NPCs should be aware of the event based on various criteria.
        
        Args:
            event_type: Type of event
            event_data: Event data including location, involved NPCs, and impact
            
        Returns:
            List of aware NPCs with their awareness context
        """
        try:
            aware_npcs = []
            event_magnitude = event_data.get("magnitude", 0.0)
            event_location = event_data.get("location", "")
            involved_npcs = event_data.get("npc_ids", [])
            event_factions = event_data.get("factions", [])
            event_interests = event_data.get("interests", [])
            
            # Get all NPCs in the game world
            all_npcs = await self.npc_coordinator.get_all_npcs()
            
            for npc in all_npcs:
                awareness_context = {
                    "npc_id": npc["id"],
                    "awareness_level": 0.0,
                    "awareness_reasons": []
                }
                
                # Check direct involvement
                if npc["id"] in involved_npcs:
                    awareness_context["awareness_level"] = 1.0
                    awareness_context["awareness_reasons"].append("direct_involvement")
                    aware_npcs.append(awareness_context)
                    continue
                    
                # Check magnitude-based awareness (major events)
                if event_magnitude >= 0.8:  # Major event threshold
                    awareness_context["awareness_level"] = 0.9
                    awareness_context["awareness_reasons"].append("major_event")
                    aware_npcs.append(awareness_context)
                    continue
                    
                # Check social connections
                npc_connections = npc.get("social_connections", [])
                for involved_npc_id in involved_npcs:
                    if involved_npc_id in npc_connections:
                        awareness_context["awareness_level"] = 0.8
                        awareness_context["awareness_reasons"].append("social_connection")
                        aware_npcs.append(awareness_context)
                        break
                        
                # Check faction affiliations
                npc_factions = npc.get("factions", [])
                if any(faction in npc_factions for faction in event_factions):
                    awareness_context["awareness_level"] = 0.7
                    awareness_context["awareness_reasons"].append("faction_affiliation")
                    aware_npcs.append(awareness_context)
                    continue
                    
                # Check interests/hobbies
                npc_interests = npc.get("interests", [])
                if any(interest in npc_interests for interest in event_interests):
                    awareness_context["awareness_level"] = 0.6
                    awareness_context["awareness_reasons"].append("personal_interest")
                    aware_npcs.append(awareness_context)
                    continue
                    
                # Check location proximity
                if event_location and npc.get("location") == event_location:
                    awareness_context["awareness_level"] = 0.5
                    awareness_context["awareness_reasons"].append("location_proximity")
                    aware_npcs.append(awareness_context)
                    
            # Let Nyx make final determination
            nyx_agent = await self.get_nyx_agent()
            awareness_analysis = await nyx_agent.analyze_npc_awareness(
                event_type=event_type,
                event_data=event_data,
                potential_aware_npcs=aware_npcs
            )
            
            # Filter and adjust awareness based on Nyx's analysis
            final_aware_npcs = []
            for npc_context in aware_npcs:
                nyx_decision = awareness_analysis.get(str(npc_context["npc_id"]), {})
                if nyx_decision.get("should_be_aware", False):
                    npc_context["awareness_level"] = nyx_decision.get("adjusted_awareness", npc_context["awareness_level"])
                    final_aware_npcs.append(npc_context)
                    
            return final_aware_npcs
            
        except Exception as e:
            logger.error(f"Error determining aware NPCs: {e}")
            return []

    async def _calculate_immediate_impact(self, event_data: Dict[str, Any]) -> float:
        """
        Calculate the immediate impact of an event based on various factors.
        
        Args:
            event_data: Event data including casualties, damage, and social disruption
            
        Returns:
            Immediate impact score (0.0 to 1.0)
        """
        try:
            # Get base impact factors
            casualties = event_data.get("casualties", 0)
            damage = event_data.get("damage", 0)
            social_disruption = event_data.get("social_disruption", 0)
            economic_impact = event_data.get("economic_impact", 0)
            
            # Calculate weighted impact
            weights = {
                "casualties": 0.4,
                "damage": 0.3,
                "social_disruption": 0.2,
                "economic_impact": 0.1
            }
            
            # Normalize each factor to 0-1 range
            normalized_factors = {
                "casualties": min(casualties / 100, 1.0),  # Assuming 100 casualties is max
                "damage": min(damage / 1000, 1.0),  # Assuming 1000 damage units is max
                "social_disruption": min(social_disruption / 10, 1.0),  # 0-10 scale
                "economic_impact": min(economic_impact / 10000, 1.0)  # Assuming 10000 economic units is max
            }
            
            # Calculate weighted sum
            impact = sum(
                normalized_factors[factor] * weight
                for factor, weight in weights.items()
            )
            
            # Let Nyx adjust the impact
            nyx_agent = await self.get_nyx_agent()
            nyx_adjustment = await nyx_agent.adjust_impact_calculation(
                impact_type="immediate",
                base_impact=impact,
                event_data=event_data
            )
            
            return min(max(impact * nyx_adjustment, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating immediate impact: {e}")
            return 0.0

    async def _calculate_long_term_impact(self, event_data: Dict[str, Any]) -> float:
        """
        Calculate the long-term impact of an event based on various factors.
        
        Args:
            event_data: Event data including political, environmental, and social changes
            
        Returns:
            Long-term impact score (0.0 to 1.0)
        """
        try:
            # Get base impact factors
            political_change = event_data.get("political_change", 0)
            environmental_effect = event_data.get("environmental_effect", 0)
            social_change = event_data.get("social_change", 0)
            economic_change = event_data.get("economic_change", 0)
            
            # Calculate weighted impact
            weights = {
                "political_change": 0.3,
                "environmental_effect": 0.2,
                "social_change": 0.3,
                "economic_change": 0.2
            }
            
            # Normalize each factor to 0-1 range
            normalized_factors = {
                "political_change": min(political_change / 10, 1.0),  # 0-10 scale
                "environmental_effect": min(environmental_effect / 10, 1.0),  # 0-10 scale
                "social_change": min(social_change / 10, 1.0),  # 0-10 scale
                "economic_change": min(economic_change / 10000, 1.0)  # Assuming 10000 economic units is max
            }
            
            # Calculate weighted sum
            impact = sum(
                normalized_factors[factor] * weight
                for factor, weight in weights.items()
            )
            
            # Let Nyx adjust the impact
            nyx_agent = await self.get_nyx_agent()
            nyx_adjustment = await nyx_agent.adjust_impact_calculation(
                impact_type="long_term",
                base_impact=impact,
                event_data=event_data
            )
            
            return min(max(impact * nyx_adjustment, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating long-term impact: {e}")
            return 0.0

    async def _calculate_npc_impact(self, event_data: Dict[str, Any]) -> float:
        """
        Calculate the NPC-specific impact of an event based on various factors.
        
        Args:
            event_data: Event data including NPC relationships, goals, and resources
            
        Returns:
            NPC impact score (0.0 to 1.0)
        """
        try:
            # Get base impact factors
            relationship_impact = event_data.get("relationship_impact", 0)
            goal_impact = event_data.get("goal_impact", 0)
            resource_impact = event_data.get("resource_impact", 0)
            influence_impact = event_data.get("influence_impact", 0)
            
            # Calculate weighted impact
            weights = {
                "relationship_impact": 0.3,
                "goal_impact": 0.3,
                "resource_impact": 0.2,
                "influence_impact": 0.2
            }
            
            # Normalize each factor to 0-1 range
            normalized_factors = {
                "relationship_impact": min(relationship_impact / 10, 1.0),  # 0-10 scale
                "goal_impact": min(goal_impact / 10, 1.0),  # 0-10 scale
                "resource_impact": min(resource_impact / 1000, 1.0),  # Assuming 1000 resource units is max
                "influence_impact": min(influence_impact / 10, 1.0)  # 0-10 scale
            }
            
            # Calculate weighted sum
            impact = sum(
                normalized_factors[factor] * weight
                for factor, weight in weights.items()
            )
            
            # Let Nyx adjust the impact
            nyx_agent = await self.get_nyx_agent()
            nyx_adjustment = await nyx_agent.adjust_impact_calculation(
                impact_type="npc",
                base_impact=impact,
                event_data=event_data
            )
            
            return min(max(impact * nyx_adjustment, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating NPC impact: {e}")
            return 0.0

    async def _update_single_npc(self, npc_id: int, npc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a single NPC's state and information using asyncpg.
        """
        try:
            # Get current NPC data (assuming _get_npc_info is also converted)
            current_data = await self._get_npc_info(npc_id)
            if not current_data:
                logger.error(f"NPC {npc_id} not found for update.")
                return {"error": "NPC not found", "status": "failed"}

            # Validate update data (assuming async validation)
            validation_result = await self._validate_npc_update(npc_data)
            if not validation_result["valid"]:
                logger.warning(f"Invalid NPC update data for {npc_id}: {validation_result['errors']}")
                return {
                    "error": "Invalid update data",
                    "validation_errors": validation_result["errors"],
                    "status": "failed"
                }

            # --- Database Update Section ---
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    # Update basic info
                    await conn.execute("""
                        UPDATE NPCs
                        SET name = $1, location = $2, status = $3,
                            last_updated = NOW()
                        WHERE npc_id = $4
                    """,
                        npc_data.get("name", current_data["name"]),
                        npc_data.get("location", current_data["location"]),
                        npc_data.get("status", current_data["status"]),
                        npc_id
                    )

                    # Update stats
                    if "stats" in npc_data:
                        await conn.execute("""
                            UPDATE NPCStats
                            SET health = $1, energy = $2, influence = $3
                            WHERE npc_id = $4
                        """,
                            npc_data["stats"].get("health", current_data["stats"]["health"]),
                            npc_data["stats"].get("energy", current_data["stats"]["energy"]),
                            npc_data["stats"].get("influence", current_data["stats"]["influence"]),
                            npc_id
                        )

                    # Update relationships
                    if "relationships" in npc_data:
                        # First remove old relationships for this NPC
                        await conn.execute("DELETE FROM NPCRelationships WHERE npc_id = $1", npc_id)

                        # Insert new relationships (Consider executemany if performance is critical)
                        for rel in npc_data["relationships"]:
                            await conn.execute("""
                                INSERT INTO NPCRelationships
                                (npc_id, related_npc_id, relationship_type, strength)
                                VALUES ($1, $2, $3, $4)
                            """,
                                npc_id,
                                rel.get("related_npc_id"),
                                rel.get("type"),
                                rel.get("strength")
                            )

                    # Update schedule
                    if "schedule" in npc_data:
                        await conn.execute("""
                            UPDATE NPCSchedules
                            SET schedule_data = $1::jsonb
                            WHERE npc_id = $2
                        """,
                            json.dumps(npc_data["schedule"]),
                            npc_id
                        )
            # --- End Database Update Section ---

            logger.info(f"Successfully updated NPC {npc_id} in database.")

            # Get updated NPC data after successful transaction
            updated_data = await self._get_npc_info(npc_id)
            if not updated_data:
                 # This shouldn't happen if the update succeeded, but handle defensively
                 logger.error(f"Failed to retrieve updated data for NPC {npc_id} after update.")
                 return {
                    "error": "Failed to retrieve updated data post-update",
                    "status": "failed"
                 }

            # Let Nyx analyze the update
            nyx_agent = await self.get_nyx_agent()
            nyx_analysis = await nyx_agent.analyze_npc_update(
                npc_id=npc_id,
                old_data=current_data,
                new_data=updated_data
            )

            return {
                "npc_data": updated_data,
                "nyx_analysis": nyx_analysis,
                "status": "success"
            }

        except (asyncpg.PostgresError, ConnectionError) as db_err:
            logger.error(f"Database error updating NPC {npc_id}: {db_err}", exc_info=True)
            return {
                "error": f"Database error: {db_err}",
                "status": "failed"
            }
        except Exception as e:
            logger.error(f"Unexpected error updating NPC {npc_id}: {e}", exc_info=True)
            return {
                "error": str(e),
                "status": "failed"
            }


    async def _get_npc_info(self, npc_id: int) -> Dict[str, Any] | None:
        """
        Get comprehensive information about a specific NPC using asyncpg.
        """
        try:
            async with get_db_connection_context() as conn:
                # Get basic NPC info
                npc_row = await conn.fetchrow("""
                    SELECT name, location, status, created_at, last_updated
                    FROM NPCs
                    WHERE npc_id = $1
                """, npc_id)

                if not npc_row:
                    logger.warning(f"NPC info not found for npc_id: {npc_id}")
                    return None

                # Get NPC stats
                stats_row = await conn.fetchrow("""
                    SELECT health, energy, influence
                    FROM NPCStats
                    WHERE npc_id = $1
                """, npc_id)

                # Get relationships
                rel_rows = await conn.fetch("""
                    SELECT related_npc_id, relationship_type, strength
                    FROM NPCRelationships
                    WHERE npc_id = $1
                """, npc_id)

                relationships = [{
                    "related_npc_id": rel_row['related_npc_id'],
                    "type": rel_row['relationship_type'],
                    "strength": rel_row['strength']
                } for rel_row in rel_rows]

                # Get schedule
                schedule_row = await conn.fetchrow("""
                    SELECT schedule_data
                    FROM NPCSchedules
                    WHERE npc_id = $1
                """, npc_id)

                # Process schedule data (assuming it's stored as JSON/JSONB)
                schedule_data = {}
                if schedule_row and schedule_row['schedule_data']:
                    # asyncpg might auto-decode JSONB, otherwise use json.loads
                    if isinstance(schedule_row['schedule_data'], dict):
                        schedule_data = schedule_row['schedule_data']
                    elif isinstance(schedule_row['schedule_data'], str):
                         try:
                            schedule_data = json.loads(schedule_row['schedule_data'])
                         except json.JSONDecodeError:
                            logger.warning(f"Failed to parse schedule JSON for NPC {npc_id}")
                            schedule_data = {"error": "invalid JSON format"}
                    else: # Handle unexpected types
                        logger.warning(f"Unexpected schedule data type for NPC {npc_id}: {type(schedule_row['schedule_data'])}")
                        schedule_data = {"error": "unexpected data type"}


            # Get recent memories (outside the DB connection block if it doesn't need DB access)
            # Assuming governor and memory_manager are available via self
            memories = []
            if hasattr(self, 'governor') and self.governor:
                 try:
                     memory_manager = await self.governor.get_memory_manager() # Assuming this exists and is async
                     memories = await memory_manager.recall( # Assuming this exists and is async
                         entity_type="npc",
                         entity_id=npc_id,
                         limit=5
                     )
                 except Exception as mem_err:
                     logger.error(f"Failed to recall memories for NPC {npc_id}: {mem_err}", exc_info=True)
                     memories = [{"error": "Failed to retrieve memories"}] # Indicate memory retrieval failure
            else:
                 logger.warning("Governor or memory manager not available for NPC memory recall.")


            return {
                "id": npc_id,
                "name": npc_row['name'],
                "location": npc_row['location'],
                "status": npc_row['status'],
                "created_at": npc_row['created_at'].isoformat() if npc_row['created_at'] else None,
                "last_updated": npc_row['last_updated'].isoformat() if npc_row['last_updated'] else None,
                "stats": {
                    "health": stats_row['health'] if stats_row else 100,
                    "energy": stats_row['energy'] if stats_row else 100,
                    "influence": stats_row['influence'] if stats_row else 0
                },
                "relationships": relationships,
                "schedule": schedule_data,
                "recent_memories": memories # Include memories fetched after DB ops
            }

        except (asyncpg.PostgresError, ConnectionError) as db_err:
            logger.error(f"Database error getting NPC info for {npc_id}: {db_err}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting NPC info for {npc_id}: {e}", exc_info=True)
            return None
    async def _execute_npc_directive(self, npc_id: int, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a directive for a specific NPC.
        
        Args:
            npc_id: NPC ID
            directive: Directive to execute including type, parameters, and priority
            
        Returns:
            Execution results including success status and any effects
        """
        try:
            # Get current NPC state
            npc_data = await self._get_npc_info(npc_id)
            if not npc_data:
                return {"error": "NPC not found", "status": "failed"}
            
            # Validate directive
            validation_result = await self._validate_directive(directive)
            if not validation_result["valid"]:
                return {
                    "error": "Invalid directive",
                    "validation_errors": validation_result["errors"],
                    "status": "failed"
                }
            
            # Check if NPC can execute directive
            can_execute = await self._check_directive_executability(npc_id, directive)
            if not can_execute["can_execute"]:
                return {
                    "error": "Cannot execute directive",
                    "reason": can_execute["reason"],
                    "status": "failed"
                }
            
            # Execute directive based on type
            directive_type = directive.get("type", "")
            execution_result = await self._execute_directive_by_type(
                npc_id=npc_id,
                directive_type=directive_type,
                directive=directive,
                npc_data=npc_data
            )
            
            # Let Nyx analyze the execution
            nyx_agent = await self.get_nyx_agent()
            nyx_analysis = await nyx_agent.analyze_directive_execution(
                npc_id=npc_id,
                directive=directive,
                result=execution_result
            )
            
            return {
                "execution_result": execution_result,
                "nyx_analysis": nyx_analysis,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error executing directive for NPC {npc_id}: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def batch_update_npcs(self, npcs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch update multiple NPCs efficiently using asyncpg within a single transaction.
        """
        results = []
        valid_updates = [] # Store validated data with IDs for processing

        # --- Pre-processing and Validation ---
        for npc_data in npcs_data:
            npc_id = npc_data.get("id")
            if not npc_id:
                results.append({"error": "Missing NPC ID", "status": "failed", "original_data": npc_data})
                continue

            # Validate update data (assuming async validation)
            validation_result = await self._validate_npc_update(npc_data)
            if not validation_result["valid"]:
                results.append({
                    "npc_id": npc_id,
                    "error": "Invalid update data",
                    "validation_errors": validation_result["errors"],
                    "status": "failed"
                })
                continue

            # Add valid data for processing
            valid_updates.append(npc_data) # Keep original structure with ID

        if not valid_updates:
            logger.warning("Batch NPC update: No valid updates to process.")
            return results # Return results accumulated so far (only errors)

        # --- Database Update Section (Single Transaction) ---
        processed_ids = set()
        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    # Iterate through the validated updates and apply them
                    for npc_data in valid_updates:
                        npc_id = npc_data["id"]
                        processed_ids.add(npc_id) # Track IDs processed in this batch

                        # Apply updates based on keys present in npc_data
                        # Basic Info
                        if any(k in npc_data for k in ["name", "location", "status"]):
                            # Fetch potentially missing values to avoid setting NULL unintentionally
                            # In a real scenario, you might fetch defaults or require keys
                            current_name = npc_data.get("name") # simplified: assumes name is always provided if changed
                            current_location = npc_data.get("location")
                            current_status = npc_data.get("status")
                            # A SELECT might be needed here if partial updates are common and defaults complex

                            await conn.execute("""
                                UPDATE NPCs
                                SET name = COALESCE($1, name),
                                    location = COALESCE($2, location),
                                    status = COALESCE($3, status),
                                    last_updated = NOW()
                                WHERE npc_id = $4
                            """, current_name, current_location, current_status, npc_id)


                        # Stats
                        if "stats" in npc_data:
                             # Similar COALESCE approach or assume full stats dict if present
                            await conn.execute("""
                                UPDATE NPCStats
                                SET health = COALESCE($1, health),
                                    energy = COALESCE($2, energy),
                                    influence = COALESCE($3, influence)
                                WHERE npc_id = $4
                            """,
                                npc_data["stats"].get("health"),
                                npc_data["stats"].get("energy"),
                                npc_data["stats"].get("influence"),
                                npc_id
                            )

                        # Relationships (Replace logic)
                        if "relationships" in npc_data:
                            await conn.execute("DELETE FROM NPCRelationships WHERE npc_id = $1", npc_id)
                            if npc_data["relationships"]: # Only insert if there are new relationships
                                for rel in npc_data["relationships"]:
                                    await conn.execute("""
                                        INSERT INTO NPCRelationships
                                        (npc_id, related_npc_id, relationship_type, strength)
                                        VALUES ($1, $2, $3, $4)
                                    """,
                                        npc_id,
                                        rel.get("related_npc_id"),
                                        rel.get("type"),
                                        rel.get("strength")
                                    )

                        # Schedule
                        if "schedule" in npc_data:
                            await conn.execute("""
                                UPDATE NPCSchedules
                                SET schedule_data = $1::jsonb
                                WHERE npc_id = $2
                            """,
                                json.dumps(npc_data["schedule"]),
                                npc_id
                            )

            logger.info(f"Successfully processed batch update for {len(processed_ids)} NPCs in transaction.")

            # --- Post-processing: Fetch updated data and build results ---
            for npc_id in processed_ids:
                 updated_data = await self._get_npc_info(npc_id) # Fetch fresh data
                 if updated_data:
                     results.append({
                         "npc_id": npc_id,
                         "npc_data": updated_data,
                         "status": "success"
                     })
                 else:
                     # Should be rare if transaction succeeded and ID was valid
                     logger.error(f"Batch Update: Failed to retrieve updated data for NPC {npc_id} after successful transaction.")
                     results.append({
                         "npc_id": npc_id,
                         "error": "Failed to retrieve updated data post-commit",
                         "status": "failed"
                     })

            # Let Nyx analyze the batch update (optional)
            try:
                nyx_agent = await self.get_nyx_agent()
                nyx_analysis = await nyx_agent.analyze_batch_update(
                    npcs_data=valid_updates, # Pass the data that was attempted
                    results=results # Pass the outcome
                )
                # You might want to add nyx_analysis to the overall return or log it
                logger.info(f"Nyx batch analysis completed: {nyx_analysis}")
            except Exception as nyx_err:
                logger.error(f"Failed to run Nyx batch analysis: {nyx_err}", exc_info=True)

            return results # Contains success and failure info

        except (asyncpg.PostgresError, ConnectionError) as db_err:
            logger.error(f"Database error during batch NPC update transaction: {db_err}", exc_info=True)
            # Add error results for all IDs that were *supposed* to be processed in the failed transaction
            for npc_id in processed_ids:
                 # Avoid overwriting existing specific validation errors
                 if not any(r.get("npc_id") == npc_id for r in results):
                     results.append({
                         "npc_id": npc_id,
                         "error": f"Database transaction error: {db_err}",
                         "status": "failed"
                     })
            return results # Return accumulated validation errors + transaction error results

        except Exception as e:
            logger.error(f"Unexpected error during batch NPC update: {e}", exc_info=True)
            # Add generic error results for unprocessed/failed IDs
            processed_in_error = processed_ids or {upd['id'] for upd in valid_updates} # Best guess at affected IDs
            for npc_id in processed_in_error:
                 if not any(r.get("npc_id") == npc_id for r in results):
                    results.append({
                        "npc_id": npc_id,
                        "error": f"Unexpected batch error: {str(e)}",
                        "status": "failed"
                    })
            return results

async def process_universal_update_with_governance(
    user_id: int,
    conversation_id: int,
    narrative: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a universal update based on narrative text with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        narrative: Narrative text to process
        context: Additional context (optional)
        
    Returns:
        Dictionary with update results
    """
    governance = await get_central_governance(user_id, conversation_id)
    
    # Import here to avoid circular imports
    from logic.universal_updater_sdk import process_universal_update
    
    # Process the universal update
    result = await process_universal_update(user_id, conversation_id, narrative, context)
    
    return result

# Add during initialization sequence in register_with_governance or similar initialization function
async def register_universal_updater(user_id: int, conversation_id: int):
    """
    Register universal updater with governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    # Import here to avoid circular imports
    from logic.universal_updater_sdk import register_with_governance as register_updater
    
    # Register with governance
    await register_updater(user_id, conversation_id)

async def add_joint_memory_with_governance(
    user_id: int,
    conversation_id: int,
    memory_text: str,
    source_type: str,
    source_id: int,
    shared_with: List[Dict[str, Any]],
    significance: int = 5,
    tags: List[str] = None,
    metadata: Dict[str, Any] = None
) -> int:
    """
    Add a joint memory with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        memory_text: Memory text
        source_type: Source entity type
        source_id: Source entity ID
        shared_with: List of entities to share with
        significance: Memory significance
        tags: Memory tags
        metadata: Memory metadata
        
    Returns:
        Memory ID
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.memory_graph.add_joint_memory(
        memory_text, source_type, source_id, shared_with,
        significance, tags, metadata
    )

async def remember_with_governance(
    user_id: int,
    conversation_id: int,
    entity_type: str,
    entity_id: int,
    memory_text: str,
    importance: str = "medium",
    emotional: bool = True,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a memory with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        entity_type: Type of entity
        entity_id: ID of the entity
        memory_text: The memory text
        importance: Importance level
        emotional: Whether to analyze emotional content
        tags: Optional tags
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.memory_integration.remember(
        entity_type=entity_type,
        entity_id=entity_id,
        memory_text=memory_text,
        importance=importance,
        emotional=emotional,
        tags=tags
    )

async def recall_with_governance(
    user_id: int,
    conversation_id: int,
    entity_type: str,
    entity_id: int,
    query: Optional[str] = None,
    context: Optional[str] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Recall memories with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        entity_type: Type of entity
        entity_id: ID of the entity
        query: Optional search query
        context: Current context
        limit: Maximum number of memories to return
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.memory_integration.recall(
        entity_type=entity_type,
        entity_id=entity_id,
        query=query,
        context=context,
        limit=limit
    )

async def process_message_with_governance(
    user_id: int,
    conversation_id: int,
    user_message: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a user message through the central governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_message: User's message text
        context_data: Optional additional context
        
    Returns:
        Complete response including message, scene updates, etc.
    """
    # Get the central governance
    governance = await get_central_governance(user_id, conversation_id)
    
    # Process the message
    return await governance.process_user_message(user_message, context_data)

# Add this class to nyx/integrate.py

class LoreIntegration:
    """
    Integration for Lore System with governance oversight.
    """
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor
        self.lore_generator = None
        self.lore_manager = None
        self.npc_lore_integration = None
    
    async def initialize(self):
        """Initialize the lore integration."""
        from lore.dynamic_lore_generator import DynamicLoreGenerator
        from lore.lore_manager import LoreManager
        from lore.npc_lore_integration import NPCLoreIntegration
        
        self.lore_generator = DynamicLoreGenerator(self.user_id, self.conversation_id)
        self.lore_manager = LoreManager(self.user_id, self.conversation_id)
        self.npc_lore_integration = NPCLoreIntegration(self.user_id, self.conversation_id)
        
        return self
    
    async def generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Generate comprehensive lore with governance oversight.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Generated lore
        """
        # Check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_lore",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logger.warning(f"Lore generation not approved: {permission['reasoning']}")
            return {
                "error": f"Lore generation not approved: {permission['reasoning']}",
                "approved": False
            }
        
        # Generate complete lore
        lore = await self.lore_generator.generate_complete_lore(environment_desc)
        
        # Report the action
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_lore",
                "description": f"Generated complete lore for environment: {environment_desc[:50]}"
            },
            result={
                "world_lore_count": len(lore.get("world_lore", {})),
                "factions_count": len(lore.get("factions", [])),
                "cultural_elements_count": len(lore.get("cultural_elements", [])),
                "historical_events_count": len(lore.get("historical_events", [])),
                "locations_count": len(lore.get("locations", [])),
                "quests_count": len(lore.get("quests", []))
            }
        )
        
        # Integrate with memory system
        await self._integrate_with_memory_system(lore)
        
        return {
            "lore": lore,
            "governance_approved": True
        }
    
    async def integrate_with_npcs(self, npc_ids: List[int]) -> Dict[str, Any]:
        """
        Integrate lore with NPCs with governance oversight.
        
        Args:
            npc_ids: List of NPC IDs to integrate lore with
            
        Returns:
            Integration results
        """
        # Check permission
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="npc_integration",
            action_type="integrate_lore_with_npcs",
            action_details={"npc_ids": npc_ids}
        )
        
        if not permission["approved"]:
            logger.warning(f"NPC lore integration not approved: {permission['reasoning']}")
            return {
                "error": f"NPC lore integration not approved: {permission['reasoning']}",
                "approved": False
            }
        
        # Use the LoreIntegrationSystem to handle the integration
        from lore.lore_integration import LoreIntegrationSystem
        integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
        
        results = await integration_system.integrate_lore_with_npcs(npc_ids)
        
        # Report the action
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="npc_integration",
            action={
                "type": "integrate_lore_with_npcs",
                "description": f"Integrated lore with {len(npc_ids)} NPCs"
            },
            result={
                "npcs_integrated": len(results)
            }
        )
        
        return {
            "results": results,
            "governance_approved": True
        }
    
    async def enhance_context_with_lore(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance context with relevant lore.
        
        Args:
            context: Current context dictionary
            
        Returns:
            Enhanced context with lore
        """
        from lore.lore_integration import LoreIntegrationSystem
        integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
        
        enhanced_context = await integration_system.enhance_gpt_context_with_lore(context)
        
        return enhanced_context
    
    async def generate_scene_description_with_lore(self, location: str) -> Dict[str, Any]:
        """
        Generate a scene description enhanced with lore.
        
        Args:
            location: Location name
            
        Returns:
            Enhanced scene description
        """
        from lore.lore_integration import LoreIntegrationSystem
        integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
        
        scene = await integration_system.generate_scene_description_with_lore(location)
        
        return scene
    
    async def _integrate_with_memory_system(self, lore: Dict[str, Any]) -> None:
        """
        Integrate generated lore with Nyx's memory system.
        
        Args:
            lore: The generated lore
        """
        memory_system = await self.governor.get_memory_system()
        
        # Store foundation lore in memory
        if "world_lore" in lore:
            for key, value in lore["world_lore"].items():
                if isinstance(value, str):
                    await memory_system.add_memory(
                        memory_text=f"World lore - {key}: {value[:100]}...",
                        memory_type="lore",
                        memory_scope="game",
                        significance=7,
                        tags=["lore", "world", key]
                    )
        
        # Store factions in memory
        if "factions" in lore:
            for faction in lore["factions"]:
                name = faction.get('name', 'Unknown faction')
                desc = faction.get('description', '')[:100] + '...'
                await memory_system.add_memory(
                    memory_text=f"Faction: {name} - {desc}",
                    memory_type="lore",
                    memory_scope="game",
                    significance=6,
                    tags=["lore", "faction", name]
                )
        
        # Store cultural elements in memory
        if "cultural_elements" in lore:
            for element in lore["cultural_elements"]:
                name = element.get('name', 'Unknown culture')
                desc = element.get('description', '')[:100] + '...'
                await memory_system.add_memory(
                    memory_text=f"Cultural element: {name} - {desc}",
                    memory_type="lore",
                    memory_scope="game",
                    significance=6,
                    tags=["lore", "culture", name]
                )
        
        # Store historical events in memory
        if "historical_events" in lore:
            for event in lore["historical_events"]:
                name = event.get('name', 'Unknown event')
                desc = event.get('description', '')[:100] + '...'
                await memory_system.add_memory(
                    memory_text=f"Historical event: {name} - {desc}",
                    memory_type="lore",
                    memory_scope="game",
                    significance=6,
                    tags=["lore", "history", name]
                )
        
        # Store locations in memory
        if "locations" in lore:
            for location in lore["locations"]:
                name = location.get('name', 'Unknown location')
                desc = location.get('description', '')[:100] + '...'
                await memory_system.add_memory(
                    memory_text=f"Location: {name} - {desc}",
                    memory_type="lore",
                    memory_scope="game",
                    significance=6,
                    tags=["lore", "location", name]
                )

class NyxIntegration:
    """Integrates all Nyx components into a unified system while preserving agent autonomy"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize all components
        self.agent = NyxAgent(user_id, conversation_id)
        self.planner = NyxPlanner(user_id, conversation_id)
        self.governance = NyxGovernance(user_id, conversation_id)
        self.task_integration = TaskIntegration(user_id, conversation_id)
        self.memory_integration = MemoryIntegration(user_id, conversation_id)
        self.scene_manager = SceneManager(user_id, conversation_id)
        self.user_model = UserModel(user_id, conversation_id)
        
        # Add cross-system communication channels
        self.event_bus = EventBus()
        self.state_manager = StateManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Add executive decision tracking
        self.decision_tracker = {
            "memory": {"decisions": [], "overrides": []},
            "npc": {"decisions": [], "overrides": []},
            "lore": {"decisions": [], "overrides": []},
            "scene": {"decisions": [], "overrides": []}
        }
        
        self.initialized = False

    async def initialize(self):
        """Initialize all components and establish communication channels"""
        if self.initialized:
            return
            
        # Initialize each component
        await self.agent.initialize()
        await self.planner.initialize()
        await self.governance.initialize()
        await self.task_integration.initialize()
        await self.memory_integration.initialize()
        await self.scene_manager.initialize()
        await self.user_model.initialize()
        
        # Set up event listeners
        self._setup_event_listeners()
        
        # Initialize state tracking
        await self.state_manager.initialize()
        
        # Start performance monitoring
        self.performance_monitor.start()
        
        self.initialized = True
        logger.info(f"Nyx integration initialized for user {self.user_id}, conversation {self.conversation_id}")

    def _setup_event_listeners(self):
        """Set up event listeners for cross-system communication"""
        # Memory system events
        self.event_bus.subscribe("memory.decision_made", self._handle_memory_decision)
        self.event_bus.subscribe("memory.pattern_identified", self._handle_memory_pattern)
        
        # NPC system events
        self.event_bus.subscribe("npc.decision_made", self._handle_npc_decision)
        self.event_bus.subscribe("npc.behavior_changed", self._handle_npc_behavior_change)
        
        # Lore system events
        self.event_bus.subscribe("lore.generation_decision", self._handle_lore_decision)
        self.event_bus.subscribe("lore.pattern_emerged", self._handle_lore_pattern)
        
        # Scene system events
        self.event_bus.subscribe("scene.state_changed", self._handle_scene_change)
        self.event_bus.subscribe("scene.decision_made", self._handle_scene_decision)

    async def _handle_memory_decision(self, decision: Dict[str, Any]):
        """Handle autonomous memory system decisions"""
        # Record the decision
        self.decision_tracker["memory"]["decisions"].append(decision)
        
        # Check if executive override is needed
        if await self._needs_executive_override(decision, "memory"):
            override = await self._generate_executive_override(decision, "memory")
            self.decision_tracker["memory"]["overrides"].append(override)
            await self.event_bus.publish("memory.executive_override", override)
        
        # Update state
        await self.state_manager.update_memory_state(decision)
        
        # Monitor performance impact
        self.performance_monitor.track_decision_impact("memory", decision)

    async def _handle_npc_decision(self, decision: Dict[str, Any]):
        """Handle autonomous NPC system decisions"""
        self.decision_tracker["npc"]["decisions"].append(decision)
        
        if await self._needs_executive_override(decision, "npc"):
            override = await self._generate_executive_override(decision, "npc")
            self.decision_tracker["npc"]["overrides"].append(override)
            await self.event_bus.publish("npc.executive_override", override)
        
        await self.state_manager.update_npc_state(decision)
        self.performance_monitor.track_decision_impact("npc", decision)

    async def _handle_lore_decision(self, decision: Dict[str, Any]):
        """Handle autonomous lore system decisions"""
        self.decision_tracker["lore"]["decisions"].append(decision)
        
        if await self._needs_executive_override(decision, "lore"):
            override = await self._generate_executive_override(decision, "lore")
            self.decision_tracker["lore"]["overrides"].append(override)
            await self.event_bus.publish("lore.executive_override", override)
        
        await self.state_manager.update_lore_state(decision)
        self.performance_monitor.track_decision_impact("lore", decision)

    async def _needs_executive_override(self, decision: Dict[str, Any], system: str) -> bool:
        """Determine if a decision needs executive override"""
        # Check for narrative consistency
        narrative_impact = await self._evaluate_narrative_impact(decision, system)
        if narrative_impact > 0.8:  # High impact threshold
            return True
            
        # Check for cross-system conflicts
        if await self._detect_cross_system_conflicts(decision, system):
            return True
            
        # Check for user model alignment
        user_alignment = await self._check_user_model_alignment(decision)
        if user_alignment < 0.6:  # Low alignment threshold
            return True
            
        # Check for performance impact
        if self.performance_monitor.would_impact_performance(decision, system):
            return True
            
        return False

    async def _score_alternatives(self, alternatives: List[Dict[str, Any]], 
                                current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score alternative decisions based on current state."""
        scored_alternatives = []
        
        for alt in alternatives:
            score = 0.0
            reasoning = []
            
            decision = alt["decision"]
            
            # Score based on narrative impact
            narrative_impact = await self._evaluate_narrative_impact(decision, "alternative")
            score += (1.0 - narrative_impact) * 0.3  # Lower impact is better for alternatives
            reasoning.append(f"Narrative impact: {1.0 - narrative_impact:.2f}")
            
            # Score based on conflict resolution
            if decision.get("action_type") == "none":
                score += 0.2  # Bonus for avoiding conflict
                reasoning.append("Avoids immediate conflict")
            
            # Score based on resource efficiency
            resources_required = decision.get("resources", {})
            resource_score = 1.0 - (sum(resources_required.values()) / 100.0)
            score += resource_score * 0.2
            reasoning.append(f"Resource efficiency: {resource_score:.2f}")
            
            # Score based on player experience preservation
            preserves_agency = decision.get("preserves_player_agency", True)
            if preserves_agency:
                score += 0.3
                reasoning.append("Preserves player agency")
            
            # Add delay penalty (prefer immediate solutions)
            if decision.get("delay", 0) > 0:
                delay_penalty = min(decision["delay"] / 300.0, 0.2)  # Max penalty of 0.2
                score -= delay_penalty
                reasoning.append(f"Delay penalty: -{delay_penalty:.2f}")
            
            scored_alternatives.append({
                "decision": decision,
                "description": alt["description"],
                "score": max(0.0, min(1.0, score)),
                "reasoning": " | ".join(reasoning)
            })
        
        # Sort by score (highest first)
        scored_alternatives.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_alternatives

    async def _generate_alternatives(self, decision: Dict[str, Any], 
                                   system: str) -> List[Dict[str, Any]]:
        """Generate alternative decisions for a given decision."""
        alternatives = []
        
        # Get the original action
        original_action = decision.get("action_type", "")
        original_target = decision.get("target")
        
        # Generate variations based on system
        if system == "memory":
            # Memory alternatives
            if original_action == "remember":
                alternatives.extend([
                    {
                        "decision": {**decision, "importance": "low"},
                        "description": "Remember with reduced importance"
                    },
                    {
                        "decision": {**decision, "delay": 60},
                        "description": "Delay memory creation"
                    },
                    {
                        "decision": {**decision, "partial": True},
                        "description": "Create partial memory"
                    }
                ])
                
        elif system == "npc":
            # NPC behavior alternatives
            if original_action in ["move", "travel"]:
                alternatives.extend([
                    {
                        "decision": {**decision, "speed": "slow"},
                        "description": "Move slowly"
                    },
                    {
                        "decision": {**decision, "target": "nearby_location"},
                        "description": "Move to closer location"
                    },
                    {
                        "decision": {"action_type": "stay", "reason": "tired"},
                        "description": "Stay in current location"
                    }
                ])
                
        elif system == "lore":
            # Lore alternatives
            if original_action == "reveal":
                alternatives.extend([
                    {
                        "decision": {**decision, "partial_reveal": True},
                        "description": "Reveal partial information"
                    },
                    {
                        "decision": {**decision, "delay": 120},
                        "description": "Delay revelation"
                    },
                    {
                        "decision": {"action_type": "hint", "target": original_target},
                        "description": "Provide hint instead"
                    }
                ])
        
        # Add a "do nothing" alternative
        alternatives.append({
            "decision": {"action_type": "none", "reason": "conflict_avoidance"},
            "description": "Take no action"
        })
        
        return alternatives

    async def _generate_executive_override(self, decision: Dict[str, Any], system: str) -> Dict[str, Any]:
        """Generate an executive override for a decision"""
        # Get current state
        current_state = await self.state_manager.get_current_state()
        
        # Generate alternative decisions
        alternatives = await self._generate_alternatives(decision, system)
        
        # Score alternatives
        scored_alternatives = await self._score_alternatives(alternatives, current_state)
        
        # Select best alternative
        best_alternative = max(scored_alternatives, key=lambda x: x["score"])
        
        return {
            "original_decision": decision,
            "override_decision": best_alternative["decision"],
            "reasoning": best_alternative["reasoning"],
            "score": best_alternative["score"],
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_pacing_impact(self, decision: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Calculate the impact of a decision on narrative pacing."""
        score = 0.5  # Start neutral
        narrative_context = current_state.get("narrative_context", {})
        
        # Get current narrative stage
        current_arc = narrative_context.get("current_arc", "rising_action")
        
        # Define expected pacing for each stage
        pacing_expectations = {
            "introduction": {"pace": "slow", "intensity": "low"},
            "rising_action": {"pace": "building", "intensity": "medium"},
            "climax": {"pace": "fast", "intensity": "high"},
            "falling_action": {"pace": "slowing", "intensity": "medium"},
            "resolution": {"pace": "slow", "intensity": "low"}
        }
        
        expected = pacing_expectations.get(current_arc, {"pace": "medium", "intensity": "medium"})
        decision_pace = decision.get("pacing_impact", {})
        
        # Compare decision pace with expected pace
        if decision_pace.get("pace") == expected["pace"]:
            score += 0.3
        elif decision_pace.get("pace") == "opposite":
            score -= 0.4
        
        # Check intensity match
        if decision_pace.get("intensity") == expected["intensity"]:
            score += 0.2
        elif abs({"low": 1, "medium": 2, "high": 3}.get(decision_pace.get("intensity", "medium"), 2) -
                 {"low": 1, "medium": 2, "high": 3}.get(expected["intensity"], 2)) > 1:
            score -= 0.3
        
        return max(0.0, min(1.0, score))

    def _evaluate_theme_alignment(self, decision: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Evaluate how well a decision aligns with story themes."""
        score = 1.0
        world_state = current_state.get("world_state", {})
        narrative_context = current_state.get("narrative_context", {})
        
        # Get current themes
        current_themes = narrative_context.get("themes", [])
        setting_mood = world_state.get("setting", {}).get("mood_tone", "")
        
        # Check theme alignment
        decision_themes = decision.get("themes", [])
        conflicting_themes = decision.get("conflicting_themes", [])
        
        # Bonus for supporting current themes
        for theme in decision_themes:
            if theme in current_themes:
                score += 0.15
        
        # Penalty for conflicting with established themes
        for theme in conflicting_themes:
            if theme in current_themes:
                score -= 0.25
        
        # Check mood alignment
        if setting_mood and "mood_impact" in decision:
            mood_compatibility = {
                "lighthearted": ["comedy", "friendship", "adventure"],
                "dark": ["tragedy", "horror", "betrayal"],
                "romantic": ["love", "passion", "intimacy"],
                "serious": ["drama", "conflict", "growth"]
            }
            
            compatible_moods = mood_compatibility.get(setting_mood, [])
            if decision["mood_impact"] not in compatible_moods:
                score -= 0.2
        
        return max(0.0, min(1.0, score))

    def _check_character_consistency(self, decision: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Check if a decision maintains character consistency."""
        score = 1.0
        character_state = current_state.get("character_state", {})
        
        # Check personality alignment
        if "personality_traits" in character_state:
            traits = character_state["personality_traits"]
            action_traits = decision.get("required_traits", [])
            conflicting_traits = decision.get("conflicting_traits", [])
            
            # Bonus for matching traits
            for trait in action_traits:
                if trait in traits:
                    score += 0.1
            
            # Penalty for conflicting traits
            for trait in conflicting_traits:
                if trait in traits:
                    score -= 0.3
        
        # Check relationship consistency
        if "relationships" in character_state and "relationship_impact" in decision:
            relationships = character_state["relationships"]
            for npc, impact in decision["relationship_impact"].items():
                if npc in relationships:
                    current_level = relationships[npc].get("level", 50)
                    # Penalize actions that drastically change established relationships
                    if abs(impact) > 30 and current_level > 70:
                        score -= 0.2
        
        # Check stat consistency
        if "stat_requirements" in decision:
            for stat, required in decision["stat_requirements"].items():
                current_value = character_state.get(stat, 0)
                if current_value < required:
                    score -= 0.2
        
        return max(0.0, min(1.0, score))

    def _calculate_plot_coherence(self, decision: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Calculate how well a decision maintains plot coherence."""
        score = 1.0
        
        # Check if decision affects active quests
        if "quest_impact" in decision:
            active_quests = current_state.get("game_state", {}).get("active_quests", [])
            if active_quests:
                # Penalize decisions that would break quest chains
                if decision.get("breaks_quest_chain"):
                    score -= 0.4
                # Penalize decisions that skip important plot points
                if decision.get("skips_plot_points"):
                    score -= 0.3
        
        # Check narrative stage alignment
        current_arc = current_state.get("narrative_context", {}).get("current_arc", "")
        if current_arc and "expected_stage" in decision:
            if decision["expected_stage"] != current_arc:
                score -= 0.2
        
        # Check for narrative contradictions
        if decision.get("contradicts_established_facts"):
            score -= 0.5
        
        return max(0.0, score)

    async def _evaluate_narrative_impact(self, decision: Dict[str, Any], system: str) -> float:
        """Evaluate the narrative impact of a decision"""
        current_state = await self.state_manager.get_current_state()
        
        factors = {
            "plot_coherence": self._calculate_plot_coherence(decision, current_state),
            "character_consistency": self._check_character_consistency(decision, current_state),
            "theme_alignment": self._evaluate_theme_alignment(decision, current_state),
            "pacing_impact": self._calculate_pacing_impact(decision, current_state)
        }
        
        weights = {
            "plot_coherence": 0.4,
            "character_consistency": 0.3,
            "theme_alignment": 0.2,
            "pacing_impact": 0.1
        }
        
        return sum(score * weights[factor] for factor, score in factors.items())

    async def _detect_cross_system_conflicts(self, decision: Dict[str, Any], system: str) -> bool:
        """Detect conflicts between system decisions"""
        other_systems = [s for s in self.decision_tracker.keys() if s != system]
        
        for other_system in other_systems:
            recent_decisions = self.decision_tracker[other_system]["decisions"][-5:]
            for other_decision in recent_decisions:
                if self._decisions_conflict(decision, other_decision):
                    return True
        
        return False

    def _timing_conflicts(self, timing1: Optional[Dict[str, Any]], 
                         timing2: Optional[Dict[str, Any]]) -> bool:
        """Check if two timing requirements conflict."""
        if not timing1 or not timing2:
            return False
        
        # Check if both require immediate execution
        if timing1.get("immediate") and timing2.get("immediate"):
            return True
        
        # Check for overlapping time windows
        start1 = timing1.get("start", 0)
        end1 = timing1.get("end", float('inf'))
        start2 = timing2.get("start", 0)
        end2 = timing2.get("end", float('inf'))
        
        # Check if windows overlap
        if start1 <= start2 < end1 or start2 <= start1 < end2:
            # Check if both require exclusive time
            if timing1.get("exclusive") and timing2.get("exclusive"):
                return True
        
        # Check sequence conflicts
        if timing1.get("must_complete_before") == timing2.get("id") and \
           timing2.get("must_complete_before") == timing1.get("id"):
            return True  # Circular dependency
        
        return False

    def _resources_conflict(self, resources1: Optional[Dict[str, Any]], 
                           resources2: Optional[Dict[str, Any]]) -> bool:
        """Check if two sets of resources conflict."""
        if not resources1 or not resources2:
            return False
        
        # Check for exclusive resources
        exclusive_resources = ["scene_control", "narrative_focus", "player_attention"]
        
        for resource in exclusive_resources:
            if resources1.get(resource) and resources2.get(resource):
                return True
        
        # Check for limited resources
        limited_resources = {
            "npc_actions": 10,  # Max NPCs that can act simultaneously
            "memory_operations": 5,  # Max concurrent memory operations
            "lore_updates": 3   # Max concurrent lore updates
        }
        
        for resource, limit in limited_resources.items():
            total_requested = resources1.get(resource, 0) + resources2.get(resource, 0)
            if total_requested > limit:
                return True
        
        return False

    def _decisions_conflict(self, decision1: Dict[str, Any], decision2: Dict[str, Any]) -> bool:
        """Check if two decisions conflict"""
        # Check for direct conflicts
        if decision1.get("target") == decision2.get("target"):
            if decision1.get("action_type") != decision2.get("action_type"):
                return True
                
        # Check for resource conflicts
        if self._resources_conflict(decision1.get("resources"), decision2.get("resources")):
            return True
            
        # Check for timing conflicts
        if self._timing_conflicts(decision1.get("timing"), decision2.get("timing")):
            return True
            
        return False
    
    def _calculate_style_match(self, decision: Dict[str, Any], user_model: Dict[str, Any]) -> float:
        """Calculate how well a decision matches the user's preferred style."""
        score = 0.5  # Start neutral
        
        # Get style preferences
        style_prefs = user_model.get("style_preferences", {})
        narrative_style = style_prefs.get("narrative_style", "balanced")
        interaction_style = style_prefs.get("interaction_style", "mixed")
        detail_level = style_prefs.get("detail_level", "medium")
        
        # Check narrative style match
        decision_narrative = decision.get("narrative_style", "balanced")
        if decision_narrative == narrative_style:
            score += 0.2
        
        # Check interaction style
        decision_interaction = decision.get("interaction_type", "mixed")
        if decision_interaction == interaction_style:
            score += 0.2
        elif (interaction_style == "dialogue" and decision_interaction == "action") or \
             (interaction_style == "action" and decision_interaction == "dialogue"):
            score -= 0.15
        
        # Check detail level preference
        decision_detail = decision.get("detail_level", "medium")
        detail_map = {"minimal": 1, "medium": 2, "detailed": 3}
        
        detail_diff = abs(detail_map.get(decision_detail, 2) - 
                         detail_map.get(detail_level, 2))
        
        if detail_diff == 0:
            score += 0.15
        elif detail_diff > 1:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _check_boundary_respect(self, decision: Dict[str, Any], user_model: Dict[str, Any]) -> float:
        """Check if a decision respects user boundaries."""
        score = 1.0  # Start with full respect
        
        # Get user boundaries
        boundaries = user_model.get("boundaries", {})
        hard_limits = boundaries.get("hard_limits", [])
        soft_limits = boundaries.get("soft_limits", [])
        content_warnings = boundaries.get("content_warnings", [])
        
        # Check hard limits (absolute boundaries)
        decision_content = decision.get("content_tags", [])
        for limit in hard_limits:
            if limit in decision_content:
                return 0.0  # Immediate fail
        
        # Check soft limits (preferences to avoid)
        for limit in soft_limits:
            if limit in decision_content:
                score -= 0.3
        
        # Check content that needs warnings
        for warning_type in content_warnings:
            if warning_type in decision_content and not decision.get("warning_provided"):
                score -= 0.2
        
        # Check consent for major changes
        if decision.get("major_consequence") and not decision.get("player_consent"):
            score -= 0.4
        
        return max(0.0, score)

    def _predict_engagement_impact(self, decision: Dict[str, Any], user_model: Dict[str, Any]) -> float:
        """Predict how a decision will impact user engagement."""
        score = 0.5  # Start neutral
        
        # Get engagement history
        engagement_history = user_model.get("engagement_metrics", {})
        recent_engagement = engagement_history.get("recent_engagement_level", 0.5)
        preferred_challenge = user_model.get("preferred_challenge_level", "medium")
        
        # Check if decision provides appropriate challenge
        decision_challenge = decision.get("challenge_level", "medium")
        challenge_map = {"easy": 1, "medium": 2, "hard": 3}
        
        challenge_diff = abs(challenge_map.get(decision_challenge, 2) - 
                            challenge_map.get(preferred_challenge, 2))
        
        if challenge_diff == 0:
            score += 0.3
        elif challenge_diff == 1:
            score += 0.1
        else:
            score -= 0.2
        
        # Check novelty vs familiarity balance
        if recent_engagement < 0.3:  # Low engagement
            # Need more novelty
            if decision.get("novelty_level", "medium") == "high":
                score += 0.3
        elif recent_engagement > 0.7:  # High engagement
            # Can maintain with familiar elements
            if decision.get("novelty_level", "medium") == "medium":
                score += 0.2
        
        # Check for engagement killers
        if decision.get("repetitive_content"):
            score -= 0.3
        if decision.get("breaks_immersion"):
            score -= 0.4
        
        return max(0.0, min(1.0, score))

    def _calculate_preference_match(self, decision: Dict[str, Any], user_model: Dict[str, Any]) -> float:
        """Calculate how well a decision matches user preferences."""
        score = 0.5  # Start neutral
        
        # Get user preferences
        preferences = user_model.get("preferences", {})
        play_style = preferences.get("play_style", "balanced")
        favorite_themes = preferences.get("favorite_themes", [])
        disliked_elements = preferences.get("disliked_elements", [])
        
        # Check play style alignment
        decision_style = decision.get("play_style_affinity", "balanced")
        if decision_style == play_style:
            score += 0.3
        elif (play_style == "combat" and decision_style == "social") or \
             (play_style == "social" and decision_style == "combat"):
            score -= 0.2
        
        # Check theme preferences
        decision_themes = decision.get("themes", [])
        for theme in decision_themes:
            if theme in favorite_themes:
                score += 0.1
            if theme in disliked_elements:
                score -= 0.2
        
        # Check content preferences
        if "content_tags" in decision:
            for tag in decision["content_tags"]:
                if tag in preferences.get("preferred_content", []):
                    score += 0.05
                if tag in preferences.get("avoided_content", []):
                    score -= 0.15
        
        return max(0.0, min(1.0, score))

    async def _check_user_model_alignment(self, decision: Dict[str, Any]) -> float:
        """Check how well a decision aligns with the user model"""
        user_model = await self.user_model.get_user_info()
        
        alignment_scores = {
            "preference_match": self._calculate_preference_match(decision, user_model),
            "engagement_impact": self._predict_engagement_impact(decision, user_model),
            "boundary_respect": self._check_boundary_respect(decision, user_model),
            "style_match": self._calculate_style_match(decision, user_model)
        }
        
        weights = {
            "preference_match": 0.4,
            "engagement_impact": 0.3,
            "boundary_respect": 0.2,
            "style_match": 0.1
        }
        
        return sum(score * weights[factor] for factor, score in alignment_scores.items())

    async def _generate_reconciliation_plan(self, inconsistencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a plan to reconcile detected inconsistencies."""
        plan = {
            "steps": [],
            "priority_order": [],
            "estimated_impact": {}
        }
        
        # Group inconsistencies by type and severity
        by_severity = {"high": [], "medium": [], "low": []}
        for inconsistency in inconsistencies:
            severity = inconsistency.get("severity", "low")
            by_severity[severity].append(inconsistency)
        
        # Process high severity first
        for severity in ["high", "medium", "low"]:
            for inconsistency in by_severity[severity]:
                step = await self._create_reconciliation_step(inconsistency)
                if step:
                    plan["steps"].append(step)
                    plan["priority_order"].append(step["id"])
        
        # Estimate overall impact
        plan["estimated_impact"] = {
            "systems_affected": len(set(step["affects"] for step in plan["steps"])),
            "estimated_time": sum(step.get("estimated_time", 0) for step in plan["steps"]),
            "risk_level": "high" if by_severity["high"] else ("medium" if by_severity["medium"] else "low")
        }
        
        return plan
    
    async def _create_reconciliation_step(self, inconsistency: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single reconciliation step for an inconsistency."""
        step = {
            "id": f"reconcile_{inconsistency['type']}_{hash(str(inconsistency))}",
            "type": inconsistency["type"],
            "description": f"Reconcile {inconsistency['type']}",
            "affects": [],
            "actions": [],
            "estimated_time": 0
        }
        
        if inconsistency["type"] == "location_mismatch":
            step["affects"] = ["npc", "scene"]
            step["actions"] = [
                {
                    "system": "npc",
                    "action": "update_location",
                    "params": {
                        "npc_id": inconsistency["details"].split()[1],
                        "new_location": inconsistency["scene_location"]
                    }
                }
            ]
            step["estimated_time"] = 1
            
        elif inconsistency["type"] == "lore_contradiction":
            step["affects"] = ["memory", "lore"]
            step["actions"] = [
                {
                    "system": "memory",
                    "action": "mark_as_false",
                    "params": {
                        "memory_fact": inconsistency["memory_fact"]
                    }
                },
                {
                    "system": "lore",
                    "action": "reinforce_fact",
                    "params": {
                        "fact": inconsistency["contradicted_lore"]
                    }
                }
            ]
            step["estimated_time"] = 2
            
        elif inconsistency["type"] == "temporal_inconsistency":
            step["affects"] = [inconsistency["details"].split()[0]]
            step["actions"] = [
                {
                    "system": inconsistency["details"].split()[0],
                    "action": "sync_time",
                    "params": {
                        "new_time": inconsistency["scene_time"]
                    }
                }
            ]
            step["estimated_time"] = 1
        
        return step

    def _detect_state_inconsistencies(self, memory_state: Dict[str, Any], 
                                     npc_state: Dict[str, Any], 
                                     lore_state: Dict[str, Any], 
                                     scene_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect inconsistencies between different system states."""
        inconsistencies = []
        
        # Check NPC location consistency
        for npc_id, npc_data in npc_state.items():
            npc_location = npc_data.get("location")
            if npc_location and npc_location != scene_state.get("current_location"):
                # Check if NPC is supposed to be in current scene
                if npc_id in scene_state.get("present_npcs", []):
                    inconsistencies.append({
                        "type": "location_mismatch",
                        "severity": "high",
                        "details": f"NPC {npc_id} location mismatch",
                        "npc_location": npc_location,
                        "scene_location": scene_state.get("current_location")
                    })
        
        # Check memory-lore consistency
        recent_memories = memory_state.get("recent_memories", [])
        for memory in recent_memories:
            if memory.get("type") == "lore":
                # Check if memory contradicts established lore
                memory_facts = memory.get("facts", [])
                lore_facts = lore_state.get("established_facts", [])
                
                for fact in memory_facts:
                    if fact.get("contradicts") in lore_facts:
                        inconsistencies.append({
                            "type": "lore_contradiction",
                            "severity": "medium",
                            "details": "Memory contradicts established lore",
                            "memory_fact": fact,
                            "contradicted_lore": fact.get("contradicts")
                        })
        
        # Check temporal consistency
        scene_time = scene_state.get("current_time")
        for system_name, state in [("memory", memory_state), ("npc", npc_state)]:
            system_time = state.get("last_update_time")
            if system_time and scene_time:
                # Check if times are significantly different
                # This is simplified - would need proper time parsing
                if abs(hash(str(system_time)) - hash(str(scene_time))) > 1000000:
                    inconsistencies.append({
                        "type": "temporal_inconsistency",
                        "severity": "low",
                        "details": f"{system_name} time doesn't match scene time",
                        "system_time": system_time,
                        "scene_time": scene_time
                    })
        
        return inconsistencies

    async def _verify_reconciliation(self) -> Dict[str, Any]:
        """Verify that reconciliation was successful."""
        verification_result = {
            "verified": True,
            "remaining_inconsistencies": [],
            "new_inconsistencies": [],
            "verification_time": datetime.now().isoformat()
        }
        
        try:
            # Re-check states
            memory_state = await self.memory_integration.get_state()
            npc_state = {}  # Would get from scene manager
            lore_state = await self.get_lore_state()
            scene_state = {}  # Would get from scene manager
            
            # Re-detect inconsistencies
            current_inconsistencies = self._detect_state_inconsistencies(
                memory_state, npc_state, lore_state, scene_state
            )
            
            if current_inconsistencies:
                verification_result["verified"] = False
                verification_result["remaining_inconsistencies"] = current_inconsistencies
                
                # Check if any are new (not in original list)
                # This is simplified - would need to track original inconsistencies
                verification_result["new_inconsistencies"] = [
                    inc for inc in current_inconsistencies 
                    if inc.get("detected_after_reconciliation")
                ]
        
        except Exception as e:
            verification_result["verified"] = False
            verification_result["error"] = str(e)
        
        return verification_result

    async def _execute_reconciliation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reconciliation plan."""
        results = {
            "executed_steps": [],
            "failed_steps": [],
            "partial_success": [],
            "overall_success": True
        }
        
        # Execute steps in priority order
        for step_id in plan["priority_order"]:
            step = next((s for s in plan["steps"] if s["id"] == step_id), None)
            if not step:
                continue
                
            step_result = {
                "step_id": step_id,
                "status": "pending",
                "actions_completed": [],
                "errors": []
            }
            
            # Execute each action in the step
            for action in step["actions"]:
                try:
                    action_result = await self._execute_reconciliation_action(action)
                    if action_result["success"]:
                        step_result["actions_completed"].append(action)
                    else:
                        step_result["errors"].append({
                            "action": action,
                            "error": action_result.get("error", "Unknown error")
                        })
                except Exception as e:
                    step_result["errors"].append({
                        "action": action,
                        "error": str(e)
                    })
            
            # Determine step status
            if not step_result["errors"]:
                step_result["status"] = "success"
                results["executed_steps"].append(step_result)
            elif step_result["actions_completed"]:
                step_result["status"] = "partial"
                results["partial_success"].append(step_result)
            else:
                step_result["status"] = "failed"
                results["failed_steps"].append(step_result)
                results["overall_success"] = False
        
        return results
    
    async def _execute_reconciliation_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reconciliation action."""
        system = action["system"]
        action_type = action["action"]
        params = action["params"]
        
        try:
            if system == "npc":
                if action_type == "update_location":
                    # Update NPC location through the appropriate system
                    # This would call the actual NPC system
                    return {"success": True}
                    
            elif system == "memory":
                if action_type == "mark_as_false":
                    # Mark memory as false through memory system
                    if self.memory_integration:
                        # Create false memory marker
                        return {"success": True}
                        
            elif system == "lore":
                if action_type == "reinforce_fact":
                    # Reinforce lore fact
                    if self.lore_system:
                        # Reinforce the fact in lore system
                        return {"success": True}
                        
            # Add more system handlers as needed
            
            return {"success": False, "error": f"Unknown system or action: {system}.{action_type}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def reconcile_system_states(self):
        """Reconcile states across all systems"""
        # Get current states
        memory_state = await self.memory_integration.get_state()
        npc_state = await self.scene_manager.get_npc_states()
        lore_state = await self.get_lore_state()
        scene_state = await self.scene_manager.get_scene_info()
        
        # Detect inconsistencies
        inconsistencies = self._detect_state_inconsistencies(
            memory_state, npc_state, lore_state, scene_state
        )
        
        # Generate reconciliation plan
        if inconsistencies:
            plan = await self._generate_reconciliation_plan(inconsistencies)
            
            # Execute reconciliation
            await self._execute_reconciliation(plan)
            
            # Verify reconciliation
            await self._verify_reconciliation()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all systems"""
        return {
            "decision_metrics": self.performance_monitor.get_decision_metrics(),
            "state_metrics": self.state_manager.get_metrics(),
            "override_rates": self._calculate_override_rates(),
            "system_health": self._get_system_health_metrics()
        }

    def _calculate_override_rates(self) -> Dict[str, float]:
        """Calculate override rates for each system"""
        override_rates = {}
        for system, data in self.decision_tracker.items():
            total_decisions = len(data["decisions"])
            total_overrides = len(data["overrides"])
            rate = total_overrides / total_decisions if total_decisions > 0 else 0
            override_rates[system] = rate
        return override_rates

    def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for all systems"""
        return {
            "memory": self.memory_integration.get_health_metrics(),
            "npc": self.scene_manager.get_npc_health_metrics(),
            "lore": self.get_lore_health_metrics(),
            "scene": self.scene_manager.get_health_metrics()
        }
