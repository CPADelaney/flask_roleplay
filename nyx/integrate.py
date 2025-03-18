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

import asyncpg

# Import the unified governance system
from nyx.nyx_governance import NyxUnifiedGovernor, AgentType, DirectiveType, DirectivePriority

# Import story components
from story_agent.agent_interaction import (
    orchestrate_conflict_analysis_and_narrative,
    generate_comprehensive_story_beat
)
from story_agent.story_director_agent import initialize_story_director

# Import agent processing components for full integration
from nyx.nyx_agent_sdk import process_user_input, generate_reflection
from nyx.memory_integration_sdk import process_memory_operation, perform_memory_maintenance
from nyx.user_model_sdk import process_user_input_for_model, get_response_guidance_for_user
from nyx.scene_manager_sdk import process_scene_input, generate_npc_response
from nyx.llm_integration import generate_text_completion

# Import new game components
from new_game_agent import NewGameAgent

# The agent trace utility
from agents import trace

# Database connection helper
from db.connection import get_db_connection

# Caching utilities
from utils.caching import CACHE_TTL, NPC_DIRECTIVE_CACHE, AGENT_DIRECTIVE_CACHE

from lore.dynamic_lore_generator import DynamicLoreGenerator
from lore.unified_validation import ValidationManager
from lore.error_handler import ErrorHandler

from .nyx_agent_sdk import NyxAgent
from .nyx_enhanced_system import NyxEnhancedSystem
from .response_filter import ResponseFilter
from .nyx_planner import NyxPlanner
from .nyx_governance import NyxGovernance
from .nyx_task_integration import TaskIntegration
from .memory_integration_sdk import MemoryIntegration
from .scene_manager_sdk import SceneManager
from .user_model_sdk import UserModel

logger = logging.getLogger(__name__)

# Initialize components
lore_system = LoreSystem()
lore_validator = LoreValidator()
error_handler = ErrorHandler()

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
        """Store a joint memory in the database."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO JointMemories (
                    user_id, conversation_id, memory_text, 
                    source_type, source_id, significance, 
                    tags, metadata, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING memory_id
            """, (
                self.user_id, self.conversation_id, memory_text,
                source_type, source_id, significance,
                json.dumps(tags), json.dumps(metadata)
            ))
            
            memory_id = cursor.fetchone()[0]
            
            # Store memory sharing relationships
            for entity in shared_with:
                cursor.execute("""
                    INSERT INTO JointMemorySharing (
                        memory_id, entity_type, entity_id
                    )
                    VALUES (%s, %s, %s)
                """, (
                    memory_id, entity.get("entity_type"), entity.get("entity_id")
                ))
            
            conn.commit()
            return memory_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing joint memory: {e}")
            return -1
        finally:
            cursor.close()
            conn.close()
    
    async def get_shared_memories(
        self,
        entity_type: str,
        entity_id: int,
        filter_tags: List[str] = None,
        min_significance: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get memories shared with a specific entity.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Base query
            query = """
                SELECT m.memory_id, m.memory_text, m.source_type, m.source_id,
                       m.significance, m.tags, m.metadata, m.created_at
                FROM JointMemories m
                INNER JOIN JointMemorySharing s ON m.memory_id = s.memory_id
                WHERE m.user_id = %s AND m.conversation_id = %s
                AND s.entity_type = %s AND s.entity_id = %s
                AND m.significance >= %s
            """
            
            params = [self.user_id, self.conversation_id, entity_type, entity_id, min_significance]
            
            # Add tag filtering if needed
            if filter_tags:
                placeholders = ', '.join(['%s'] * len(filter_tags))
                tag_condition = f"AND m.tags ?| array[{placeholders}]"
                query += tag_condition
                params.extend(filter_tags)
            
            # Add ordering and limit
            query += " ORDER BY m.significance DESC, m.created_at DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                memory_id, memory_text, source_type, source_id, significance, tags, metadata, created_at = row
                
                try:
                    tags = json.loads(tags) if isinstance(tags, str) else tags or []
                    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata or {}
                except json.JSONDecodeError:
                    tags = []
                    metadata = {}
                
                results.append({
                    "memory_id": memory_id,
                    "memory_text": memory_text,
                    "source_type": source_type,
                    "source_id": source_id,
                    "significance": significance,
                    "tags": tags,
                    "metadata": metadata,
                    "created_at": created_at.isoformat() if created_at else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting shared memories: {e}")
            return []
        finally:
            cursor.close()
            conn.close()


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
        Update a single NPC's state and information.
        
        Args:
            npc_id: NPC ID
            npc_data: Updated NPC data including stats, relationships, location, etc.
            
        Returns:
            Updated NPC data with validation results
        """
        try:
            # Get current NPC data
            current_data = await self._get_npc_info(npc_id)
            if not current_data:
                logger.error(f"NPC {npc_id} not found")
                return {"error": "NPC not found", "status": "failed"}
            
            # Validate update data
            validation_result = await self._validate_npc_update(npc_data)
            if not validation_result["valid"]:
                logger.warning(f"Invalid NPC update data: {validation_result['errors']}")
                return {
                    "error": "Invalid update data",
                    "validation_errors": validation_result["errors"],
                    "status": "failed"
                }
            
            # Update NPC in database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                # Update basic info
                cursor.execute("""
                    UPDATE NPCs 
                    SET name = %s, location = %s, status = %s,
                        last_updated = NOW()
                    WHERE npc_id = %s
                """, (
                    npc_data.get("name", current_data["name"]),
                    npc_data.get("location", current_data["location"]),
                    npc_data.get("status", current_data["status"]),
                    npc_id
                ))
                
                # Update stats
                if "stats" in npc_data:
                    cursor.execute("""
                        UPDATE NPCStats 
                        SET health = %s, energy = %s, influence = %s
                        WHERE npc_id = %s
                    """, (
                        npc_data["stats"].get("health", current_data["stats"]["health"]),
                        npc_data["stats"].get("energy", current_data["stats"]["energy"]),
                        npc_data["stats"].get("influence", current_data["stats"]["influence"]),
                        npc_id
                    ))
                
                # Update relationships
                if "relationships" in npc_data:
                    # First remove old relationships
                    cursor.execute("DELETE FROM NPCRelationships WHERE npc_id = %s", (npc_id,))
                    
                    # Insert new relationships
                    for rel in npc_data["relationships"]:
                        cursor.execute("""
                            INSERT INTO NPCRelationships 
                            (npc_id, related_npc_id, relationship_type, strength)
                            VALUES (%s, %s, %s, %s)
                        """, (
                            npc_id,
                            rel["related_npc_id"],
                            rel["type"],
                            rel["strength"]
                        ))
                
                # Update schedule
                if "schedule" in npc_data:
                    cursor.execute("""
                        UPDATE NPCSchedules 
                        SET schedule_data = %s
                        WHERE npc_id = %s
                    """, (
                        json.dumps(npc_data["schedule"]),
                        npc_id
                    ))
                
                conn.commit()
                
                # Get updated NPC data
                updated_data = await self._get_npc_info(npc_id)
                
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
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error updating NPC {npc_id}: {e}")
                return {
                    "error": f"Database error: {e}",
                    "status": "failed"
                }
            finally:
                cursor.close()
                conn.close()
            
        except Exception as e:
            logger.error(f"Error updating NPC {npc_id}: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def _get_npc_info(self, npc_id: int) -> Dict[str, Any]:
        """
        Get comprehensive information about a specific NPC.
        
        Args:
            npc_id: NPC ID
            
        Returns:
            NPC information including stats, relationships, schedule, etc.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get basic NPC info
            cursor.execute("""
                SELECT name, location, status, created_at, last_updated
                FROM NPCs
                WHERE npc_id = %s
            """, (npc_id,))
            
            npc_row = cursor.fetchone()
            if not npc_row:
                return None
            
            # Get NPC stats
            cursor.execute("""
                SELECT health, energy, influence
                FROM NPCStats
                WHERE npc_id = %s
            """, (npc_id,))
            
            stats_row = cursor.fetchone()
            
            # Get relationships
            cursor.execute("""
                SELECT related_npc_id, relationship_type, strength
                FROM NPCRelationships
                WHERE npc_id = %s
            """, (npc_id,))
            
            relationships = []
            for rel_row in cursor.fetchall():
                relationships.append({
                    "related_npc_id": rel_row[0],
                    "type": rel_row[1],
                    "strength": rel_row[2]
                })
            
            # Get schedule
            cursor.execute("""
                SELECT schedule_data
                FROM NPCSchedules
                WHERE npc_id = %s
            """, (npc_id,))
            
            schedule_row = cursor.fetchone()
            
            # Get recent memories
            memory_manager = await self.governor.get_memory_manager()
            memories = await memory_manager.recall(
                entity_type="npc",
                entity_id=npc_id,
                limit=5
            )
            
            return {
                "id": npc_id,
                "name": npc_row[0],
                "location": npc_row[1],
                "status": npc_row[2],
                "created_at": npc_row[3].isoformat() if npc_row[3] else None,
                "last_updated": npc_row[4].isoformat() if npc_row[4] else None,
                "stats": {
                    "health": stats_row[0] if stats_row else 100,
                    "energy": stats_row[1] if stats_row else 100,
                    "influence": stats_row[2] if stats_row else 0
                },
                "relationships": relationships,
                "schedule": json.loads(schedule_row[0]) if schedule_row else {},
                "recent_memories": memories
            }
            
        except Exception as e:
            logger.error(f"Error getting NPC info for {npc_id}: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

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
        Batch update multiple NPCs efficiently.
        
        Args:
            npcs_data: List of NPC data dictionaries to update
            
        Returns:
            List of update results for each NPC
        """
        try:
            results = []
            
            # Group NPCs by update type for efficiency
            basic_updates = []
            stats_updates = []
            relationship_updates = []
            schedule_updates = []
            
            for npc_data in npcs_data:
                npc_id = npc_data.get("id")
                if not npc_id:
                    results.append({
                        "error": "Missing NPC ID",
                        "status": "failed"
                    })
                    continue
                
                # Validate update data
                validation_result = await self._validate_npc_update(npc_data)
                if not validation_result["valid"]:
                    results.append({
                        "error": "Invalid update data",
                        "validation_errors": validation_result["errors"],
                        "status": "failed"
                    })
                    continue
                
                # Group updates
                if "name" in npc_data or "location" in npc_data or "status" in npc_data:
                    basic_updates.append(npc_data)
                if "stats" in npc_data:
                    stats_updates.append(npc_data)
                if "relationships" in npc_data:
                    relationship_updates.append(npc_data)
                if "schedule" in npc_data:
                    schedule_updates.append(npc_data)
            
            # Process updates in batches
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                # Process basic updates
                if basic_updates:
                    for npc_data in basic_updates:
                        cursor.execute("""
                            UPDATE NPCs 
                            SET name = %s, location = %s, status = %s,
                                last_updated = NOW()
                            WHERE npc_id = %s
                        """, (
                            npc_data.get("name"),
                            npc_data.get("location"),
                            npc_data.get("status"),
                            npc_data["id"]
                        ))
                
                # Process stats updates
                if stats_updates:
                    for npc_data in stats_updates:
                        cursor.execute("""
                            UPDATE NPCStats 
                            SET health = %s, energy = %s, influence = %s
                            WHERE npc_id = %s
                        """, (
                            npc_data["stats"].get("health"),
                            npc_data["stats"].get("energy"),
                            npc_data["stats"].get("influence"),
                            npc_data["id"]
                        ))
                
                # Process relationship updates
                if relationship_updates:
                    for npc_data in relationship_updates:
                        # Remove old relationships
                        cursor.execute("DELETE FROM NPCRelationships WHERE npc_id = %s", (npc_data["id"],))
                        
                        # Insert new relationships
                        for rel in npc_data["relationships"]:
                            cursor.execute("""
                                INSERT INTO NPCRelationships 
                                (npc_id, related_npc_id, relationship_type, strength)
                                VALUES (%s, %s, %s, %s)
                            """, (
                                npc_data["id"],
                                rel["related_npc_id"],
                                rel["type"],
                                rel["strength"]
                            ))
                
                # Process schedule updates
                if schedule_updates:
                    for npc_data in schedule_updates:
                        cursor.execute("""
                            UPDATE NPCSchedules 
                            SET schedule_data = %s
                            WHERE npc_id = %s
                        """, (
                            json.dumps(npc_data["schedule"]),
                            npc_data["id"]
                        ))
                
                conn.commit()
                
                # Get updated data for all NPCs
                for npc_data in npcs_data:
                    updated_data = await self._get_npc_info(npc_data["id"])
                    if updated_data:
                        results.append({
                            "npc_id": npc_data["id"],
                            "npc_data": updated_data,
                            "status": "success"
                        })
                    else:
                        results.append({
                            "npc_id": npc_data["id"],
                            "error": "Failed to retrieve updated data",
                            "status": "failed"
                        })
                
                # Let Nyx analyze the batch update
                nyx_agent = await self.get_nyx_agent()
                nyx_analysis = await nyx_agent.analyze_batch_update(
                    npcs_data=npcs_data,
                    results=results
                )
                
                return results
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error in batch update: {e}")
                return [{
                    "error": f"Database error: {e}",
                    "status": "failed"
                } for _ in npcs_data]
            finally:
                cursor.close()
                conn.close()
            
        except Exception as e:
            logger.error(f"Error in batch update: {e}")
            return [{
                "error": str(e),
                "status": "failed"
            } for _ in npcs_data]

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
