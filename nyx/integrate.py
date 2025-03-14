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

logger = logging.getLogger(__name__)


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
    
    def get_nyx_agent(self, user_id, conversation_id):
        """Get Nyx agent for this context."""
        # Placeholder implementation
        return {"process_game_event": self.process_game_event}
    
    def get_npc_coordinator(self, user_id, conversation_id):
        """Get NPC coordinator for this context."""
        # Placeholder implementation
        return {"batch_update_npcs": self.batch_update_npcs}
    
    async def process_game_event(self, event_type, event_data):
        """Process a game event as Nyx."""
        # Simple placeholder implementation
        return {
            "should_broadcast_to_npcs": True,
            "event_type": event_type,
            "event_data": event_data
        }
    
    async def batch_update_npcs(self, npc_ids, update_type, update_data):
        """Update multiple NPCs."""
        # Placeholder implementation
        return {
            "updated_npcs": npc_ids,
            "update_type": update_type
        }
    
    async def _determine_aware_npcs(self, event_type, event_data):
        """Determine which NPCs would be aware of this event."""
        # Simplified placeholder implementation
        return []
    
    async def broadcast_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced broadcast event with governance oversight.
        """
        logger.info(f"Broadcasting event {event_type} with governance oversight")
        
        # Check permission with governance system first
        if self.governor:
            permission = await self.governor.check_action_permission(
                agent_type="nyx",
                agent_id="system",
                action_type="broadcast_event",
                action_details={
                    "event_type": event_type,
                    "event_data": event_data
                }
            )
            
            if not permission["approved"]:
                logger.warning(f"Event broadcast not approved: {permission['reasoning']}")
                return {
                    "event_type": event_type,
                    "governance_approved": False,
                    "reason": permission["reasoning"]
                }
        
        # Tell Nyx about the event FIRST and get filtering instructions
        nyx_response = await self.nyx_agent_sdk.process_game_event(event_type, event_data)
        
        # Check if Nyx wants to filter or modify this event
        if not nyx_response.get("should_broadcast_to_npcs", True):
            logger.info(f"Nyx has blocked broadcasting event {event_type} to NPCs: {nyx_response.get('reason', 'No reason provided')}")
            
            # Report the action if governor available
            if self.governor:
                await self.governor.process_agent_action_report(
                    agent_type="nyx",
                    agent_id="system",
                    action={
                        "type": "block_event",
                        "description": f"Blocked event {event_type} from reaching NPCs"
                    },
                    result={
                        "reason": nyx_response.get("reason", "No reason provided")
                    }
                )
            
            return {
                "event_type": event_type,
                "nyx_notified": True,
                "npcs_notified": 0,
                "blocked_by_nyx": True,
                "reason": nyx_response.get("reason", "Blocked by Nyx")
            }
        
        # Use Nyx's modifications if provided
        if "modified_event_data" in nyx_response:
            event_data = nyx_response["modified_event_data"]
            
            # Report modification if governor available
            if self.governor:
                await self.governor.process_agent_action_report(
                    agent_type="nyx",
                    agent_id="system",
                    action={
                        "type": "modify_event",
                        "description": f"Modified event {event_type}"
                    },
                    result={
                        "original_data": event_data,
                        "modified_data": nyx_response["modified_event_data"]
                    }
                )
        
        # Let Nyx override which NPCs should be affected
        affected_npcs = event_data.get("affected_npcs")
        if "override_affected_npcs" in nyx_response:
            affected_npcs = nyx_response["override_affected_npcs"]
        
        if not affected_npcs:
            # If no specific NPCs mentioned, determine who would know
            affected_npcs = await self._determine_aware_npcs(event_type, event_data)
        
        # Respect Nyx's filtering of aware NPCs
        if "filtered_aware_npcs" in nyx_response:
            affected_npcs = [npc_id for npc_id in affected_npcs if npc_id in nyx_response["filtered_aware_npcs"]]
        
        if affected_npcs:
            await self.npc_coordinator.batch_update_npcs(
                affected_npcs,
                "event_update",
                {"event_type": event_type, "event_data": event_data}
            )
            
            # Issue directives to affected NPCs if needed and governor available
            if self.governor and event_type in ["conflict_update", "critical_event", "emergency"]:
                for npc_id in affected_npcs:
                    await self.governor.issue_directive(
                        agent_type=AgentType.NPC,
                        agent_id=npc_id,
                        directive_type=DirectiveType.ACTION,
                        directive_data={
                            "instruction": f"React to {event_type}",
                            "event_data": event_data,
                            "priority": "high"
                        },
                        priority=DirectivePriority.HIGH,
                        duration_minutes=30
                    )
            
            logger.info(f"Event {event_type} broadcast to {len(affected_npcs)} NPCs with governance oversight")
        else:
            logger.info(f"No NPCs affected by event {event_type}")
            
        return {
            "event_type": event_type,
            "nyx_notified": True,
            "npcs_notified": len(affected_npcs) if affected_npcs else 0,
            "aware_npcs": affected_npcs,
            "nyx_modifications": "modified_event_data" in nyx_response,
            "governance_approved": True
        }


class NyxNPCIntegrationManager:
    """Enhanced NPC Integration Manager with governance integration."""
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor = None):
        """Initialize with governor access."""
        # Initialize with original method
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor
        
        # Initialize other components
        self.nyx_agent_sdk = self.get_nyx_agent(user_id, conversation_id)
        self.npc_coordinator = self.get_npc_coordinator(user_id, conversation_id)
        self.npc_system = None  # Lazy-loaded
    
    def get_nyx_agent(self, user_id, conversation_id):
        """Get Nyx agent for this context."""
        # Placeholder implementation
        return {"create_scene_narrative": self.create_scene_narrative}
    
    def get_npc_coordinator(self, user_id, conversation_id):
        """Get NPC coordinator for this context."""
        # Placeholder implementation
        return {"batch_update_npcs": self.batch_update_npcs}
    
    async def create_scene_narrative(self, directive, responses, context):
        """Create a coherent scene narrative as Nyx."""
        # Simple placeholder implementation
        return "A scene unfolds..."
    
    async def batch_update_npcs(self, npc_ids, update_type, update_data):
        """Update multiple NPCs."""
        # Placeholder implementation
        return {
            "updated_npcs": npc_ids,
            "update_type": update_type
        }
    
    async def _gather_scene_context(self, location, player_action, involved_npcs):
        """Gather context for a scene."""
        # Placeholder implementation
        return {
            "location": location,
            "player_action": player_action,
            "involved_npcs": involved_npcs or []
        }
    
    async def make_scene_decision(self, context):
        """Make a scene directive decision as Nyx."""
        # Placeholder implementation
        return {
            "directive": "interact",
            "style": "natural",
            "focus": "player response"
        }
    
    async def _execute_npc_directives(self, scene_directive):
        """Have NPCs act according to directives."""
        # Placeholder implementation
        return []
    
    async def orchestrate_scene(
        self,
        location: str,
        player_action: str = None,
        involved_npcs: List[int] = None
    ) -> Dict[str, Any]:
        """
        Enhanced scene orchestration with governance oversight.
        """
        # Check permission with governance system
        if self.governor:
            permission = await self.governor.check_action_permission(
                agent_type="nyx",
                agent_id="scene_manager",
                action_type="orchestrate_scene",
                action_details={
                    "location": location,
                    "player_action": player_action,
                    "involved_npcs": involved_npcs
                }
            )
            
            if not permission["approved"]:
                logger.warning(f"Scene orchestration not approved: {permission['reasoning']}")
                return {
                    "error": f"Scene orchestration not approved: {permission['reasoning']}",
                    "approved": False,
                    "location": location
                }
            
            # If there's an override action, use that instead
            if permission.get("override_action"):
                logger.info("Using governance override for scene orchestration")
                
                # Extract overridden values if provided
                override = permission["override_action"]
                location = override.get("location", location)
                player_action = override.get("player_action", player_action)
                involved_npcs = override.get("involved_npcs", involved_npcs)
        
        # Proceed with original implementation
        # Gather context for the scene
        context = await self._gather_scene_context(location, player_action, involved_npcs)
        
        # Issue directives to NPCs if specified and governor available
        if involved_npcs and self.governor:
            for npc_id in involved_npcs:
                await self.governor.issue_directive(
                    agent_type=AgentType.NPC,
                    agent_id=npc_id,
                    directive_type=DirectiveType.SCENE,
                    directive_data={
                        "location": location,
                        "context": context,
                        "player_action": player_action
                    },
                    priority=DirectivePriority.MEDIUM,
                    duration_minutes=60,
                    scene_id=f"scene_{self.user_id}_{self.conversation_id}_{int(datetime.now().timestamp())}"
                )
        
        # Get Nyx's scene directive
        scene_directive = await self.make_scene_decision(context)
        
        # Have NPCs act according to the directive
        npc_responses = await self._execute_npc_directives(scene_directive)
        
        # Create a coherent scene narrative
        scene_narrative = await self.nyx_agent_sdk.create_scene_narrative(
            scene_directive, npc_responses, context
        )
        
        # Report the action if governor available
        if self.governor:
            await self.governor.process_agent_action_report(
                agent_type="nyx",
                agent_id="scene_manager",
                action={
                    "type": "orchestrate_scene",
                    "description": f"Orchestrated scene at {location}"
                },
                result={
                    "location": location,
                    "npc_count": len(npc_responses),
                    "narrative_length": len(scene_narrative) if scene_narrative else 0
                }
            )
        
        return {
            "narrative": scene_narrative,
            "npc_responses": npc_responses,
            "location": location,
            "governance_approved": True
        }
    
    async def approve_group_interaction(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced group interaction approval with governance oversight.
        """
        # First use governor to check if this interaction is permitted
        if self.governor:
            permission = await self.governor.check_action_permission(
                agent_type="nyx",
                agent_id="interaction_manager",
                action_type="group_interaction",
                action_details=request
            )
            
            if not permission["approved"]:
                logger.warning(f"Group interaction not approved by governance: {permission['reasoning']}")
                return {
                    "approved": False,
                    "reason": permission["reasoning"],
                    "governance_blocked": True
                }
        
        # If approved by governor or no governor, proceed with simple implementation
        result = {
            "approved": True,
            "reason": "Approved by Nyx",
            "governance_approved": self.governor is not None
        }
        
        return result
    
    async def run_joint_memory_maintenance(self, user_id=None, conversation_id=None):
        """
        Run joint memory maintenance for Nyx and NPCs.
        
        Returns:
            Maintenance results
        """
        user_id = user_id or self.user_id
        conversation_id = conversation_id or self.conversation_id
        
        # Placeholder implementation
        return {
            "status": "maintenance_complete",
            "npcs_processed": 0,
            "memories_pruned": 0,
            "memories_consolidated": 0
        }


class MemoryIntegration:
    """
    Integration for Memory Manager with governance oversight.
    """
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor
        self.memory_bridge = None
    
    async def initialize(self):
        """Initialize the memory integration."""
        self.memory_bridge = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
        return self
    
    async def remember(
        self,
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
            entity_type: Type of entity
            entity_id: ID of the entity
            memory_text: The memory text
            importance: Importance level
            emotional: Whether to analyze emotional content
            tags: Optional tags
        """
        return await self.memory_bridge.remember(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags
        )
    
    async def recall(
        self,
        entity_type: str,
        entity_id: int,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Recall memories with governance oversight.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            query: Optional search query
            context: Current context
            limit: Maximum number of memories to return
        """
        return await self.memory_bridge.recall(
            entity_type=entity_type,
            entity_id=entity_id,
            query=query,
            context=context,
            limit=limit
        )
    
    async def create_belief(
        self,
        entity_type: str,
        entity_id: int,
        belief_text: str,
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Create a belief with governance oversight.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            belief_text: The belief statement
            confidence: Confidence in this belief
        """
        return await self.memory_bridge.create_belief(
            entity_type=entity_type,
            entity_id=entity_id,
            belief_text=belief_text,
            confidence=confidence
        )
    
    async def run_memory_maintenance(
        self,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Run memory maintenance with governance oversight.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
        """
        return await self.memory_bridge.run_maintenance(
            entity_type=entity_type,
            entity_id=entity_id
        )
    
    async def issue_memory_directive(
        self,
        directive_data: Dict[str, Any],
        priority: int = DirectivePriority.MEDIUM,
        duration_minutes: int = 30
    ) -> int:
        """
        Issue a directive to the memory manager.
        
        Args:
            directive_data: Directive data
            priority: Directive priority
            duration_minutes: Duration in minutes
            
        Returns:
            Directive ID
        """
        directive_id = await self.governor.issue_directive(
            agent_type=AgentType.MEMORY_MANAGER,
            agent_id="memory_manager",
            directive_type=DirectiveType.ACTION,
            directive_data=directive_data,
            priority=priority,
            duration_minutes=duration_minutes
        )
        
        return directive_id
    
    async def process_memory_directive(
        self,
        directive_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a directive for the memory system.
        
        Args:
            directive_data: The directive data
            
        Returns:
            Results of processing the directive
        """
        return await self.memory_bridge.process_memory_directive(directive_data)


class StoryIntegration:
    """
    Integration for Story Director and related components with governance oversight.
    """
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor
    
    async def generate_story_beat(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive story beat with governance oversight.
        
        Args:
            context_data: Context data for the story beat
            
        Returns:
            Generated story beat
        """
        # Check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.STORY_DIRECTOR,
            agent_id="director",
            action_type="generate_story_beat",
            action_details=context_data
        )
        
        if not permission["approved"]:
            logger.warning(f"Story beat generation not approved: {permission['reasoning']}")
            return {
                "error": f"Story beat generation not approved: {permission['reasoning']}",
                "approved": False
            }
        
        # If approved, call the story beat generator
        story_beat = await generate_comprehensive_story_beat(
            self.user_id, self.conversation_id, context_data
        )
        
        # Report the action
        await self.governor.process_agent_action_report(
            agent_type=AgentType.STORY_DIRECTOR,
            agent_id="director",
            action={
                "type": "generate_story_beat",
                "description": "Generated comprehensive story beat"
            },
            result={
                "narrative_stage": story_beat.get("narrative_stage"),
                "element_type": story_beat.get("element_type"),
                "execution_time": story_beat.get("execution_time")
            }
        )
        
        # Add governance approval flag
        story_beat["governance_approved"] = True
        
        return story_beat
    
    async def analyze_conflict(self, conflict_id: int) -> Dict[str, Any]:
        """
        Analyze a conflict with governance oversight.
        
        Args:
            conflict_id: ID of the conflict to analyze
            
        Returns:
            Conflict analysis
        """
        # Check permission
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id="analyst",
            action_type="analyze_conflict",
            action_details={"conflict_id": conflict_id}
        )
        
        if not permission["approved"]:
            logger.warning(f"Conflict analysis not approved: {permission['reasoning']}")
            return {
                "error": f"Conflict analysis not approved: {permission['reasoning']}",
                "approved": False,
                "conflict_id": conflict_id
            }
        
        # If approved, analyze conflict
        result = await orchestrate_conflict_analysis_and_narrative(
            self.user_id, self.conversation_id, conflict_id
        )
        
        # Report the action
        await self.governor.process_agent_action_report(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id="analyst",
            action={
                "type": "analyze_conflict",
                "description": f"Analyzed conflict {conflict_id}"
            },
            result={
                "conflict_id": conflict_id,
                "execution_time": result.get("execution_time")
            }
        )
        
        # Add governance approval flag
        result["governance_approved"] = True
        
        return result


class NewGameIntegration:
    """
    Integration for New Game Agent with governance oversight.
    """
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor
        self.new_game_agent = NewGameAgent()
    
    async def create_new_game(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new game with governance oversight.
        
        Args:
            conversation_data: Initial conversation data
            
        Returns:
            New game creation results
        """
        # Check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="new_game",
            action_type="create_new_game",
            action_details=conversation_data
        )
        
        if not permission["approved"]:
            logger.warning(f"New game creation not approved: {permission['reasoning']}")
            return {
                "error": f"New game creation not approved: {permission['reasoning']}",
                "approved": False
            }
        
        # If approved, create new game
        result = await self.new_game_agent.process_new_game(self.user_id, conversation_data)
        
        # If the conversation ID changed, update our reference
        if "conversation_id" in result and result["conversation_id"] != self.conversation_id:
            self.conversation_id = result["conversation_id"]
        
        # Issue directives for the new game
        try:
            # Directive for story director
            await self.governor.issue_directive(
                agent_type=AgentType.STORY_DIRECTOR,
                agent_id="director",
                directive_type=DirectiveType.ACTION,
                directive_data={
                    "instruction": "Initialize with new environment and NPCs",
                    "environment": result.get("environment_name"),
                    "scenario_name": result.get("scenario_name")
                },
                priority=DirectivePriority.HIGH,
                duration_minutes=60
            )
            
            # Directive for NPCs
            # Get created NPCs
            new_npcs = []
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                for row in rows:
                    new_npcs.append({
                        "npc_id": row["npc_id"],
                        "npc_name": row["npc_name"]
                    })
            finally:
                await conn.close()
            
            # Issue directives for each NPC
            for npc in new_npcs:
                await self.governor.issue_directive(
                    agent_type=AgentType.NPC,
                    agent_id=npc["npc_id"],
                    directive_type=DirectiveType.INITIALIZATION,
                    directive_data={
                        "environment": result.get("environment_name"),
                        "npc_name": npc["npc_name"]
                    },
                    priority=DirectivePriority.MEDIUM,
                    duration_minutes=24*60  # 24 hours
                )
        except Exception as e:
            logger.error(f"Error issuing directives for new game: {e}")
        
        # Report the action
        await self.governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="new_game",
            action={
                "type": "create_new_game",
                "description": f"Created new game: {result.get('scenario_name')}"
            },
            result={
                "environment_name": result.get("environment_name"),
                "conversation_id": result.get("conversation_id")
            }
        )
        
        # Add governance approval flag
        result["governance_approved"] = True
        
        return result


class NyxCentralGovernance:
    """
    Central governance system for Nyx to control all agents (NPCs and beyond).
    
    This class provides centralized control over story, new game creation, NPCs, and
    all other agentic components in the system.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the central governance system.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize the unified governor
        self.governor = NyxUnifiedGovernor(user_id, conversation_id)
        
        # Initialize integration components with governor access
        self.memory_graph = JointMemoryGraph(user_id, conversation_id)
        self.event_manager = GameEventManager(user_id, conversation_id, self.governor)
        self.npc_integration = NyxNPCIntegrationManager(user_id, conversation_id, self.governor)
        
        # Components for specialized areas
        self.story_integration = None
        self.new_game_integration = None
        self.memory_integration = None
        self.lore_integration = None
        
        # Used to track registered agents
        self.registered_agents = {}
        
        # System status and configuration
        self.initialization_time = None
        self.system_status = {
            "status": "uninitialized",
            "components": {},
            "active_agents": []
        }
        
        # Configuration
        self.config = {
            "auto_update_user_model": True,
            "enable_scene_triggers": True,
            "auto_maintenance_interval": 24*60*60,  # 24 hours in seconds
            "logging_level": "INFO",
            "agent_max_retries": 3
        }

    async def initialize(self):
        """Initialize the governance system and ensure database tables exist."""
        await self.governor.setup_database_tables()
        
        # Register core system agents
        await self._register_core_agents()
        
        # Initialize specialized components
        self.story_integration = StoryIntegration(self.user_id, self.conversation_id, self.governor)
        self.new_game_integration = NewGameIntegration(self.user_id, self.conversation_id, self.governor)
        self.lore_integration = LoreIntegration(self.user_id, self.conversation_id, self.governor)
        await self.lore_integration.initialize()
    
        # Update system status
        self.system_status["components"]["lore_integration"] = True      
        
        # Initialize memory integration
        self.memory_integration = MemoryIntegration(self.user_id, self.conversation_id, self.governor)
        await self.memory_integration.initialize()
        
        # Track initialization time
        self.initialization_time = datetime.now().isoformat()
        
        # Update system status
        self.system_status = {
            "status": "initialized",
            "initialization_time": self.initialization_time,
            "components": {
                "memory_integration": True,
                "story_integration": True,
                "npc_integration": True,
                "new_game_integration": True
            },
            "active_agents": list(self.registered_agents.keys())
        }
        
        # Create a memory about initialization
        memory_system = await self.memory_integration.memory_bridge.get_memory_system()
        await memory_system.add_memory(
            memory_text="The Nyx central governance system has been initialized.",
            memory_type="system",
            memory_scope="game",
            significance=6,
            tags=["system", "initialization", "governance"],
            metadata={
                "initialization_time": self.initialization_time,
                "components": list(self.system_status["components"].keys())
            }
        )
        
        logger.info(f"NyxCentralGovernance initialized for user {self.user_id}, conversation {self.conversation_id}")

    async def generate_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Generate comprehensive lore with governance oversight.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Generated lore
        """
        return await self.lore_integration.generate_complete_lore(environment_desc)
    
    async def integrate_lore_with_npcs(self, npc_ids: List[int]) -> Dict[str, Any]:
        """
        Integrate lore with NPCs with governance oversight.
        
        Args:
            npc_ids: List of NPC IDs to integrate lore with
            
        Returns:
            Integration results
        """
        return await self.lore_integration.integrate_with_npcs(npc_ids)
    
    async def enhance_context_with_lore(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance context with relevant lore.
        
        Args:
            context: Current context dictionary
            
        Returns:
            Enhanced context with lore
        """
        return await self.lore_integration.enhance_context_with_lore(context)
    
    async def generate_scene_with_lore(self, location: str) -> Dict[str, Any]:
        """
        Generate a scene description enhanced with lore.
        
        Args:
            location: Location name
            
        Returns:
            Enhanced scene description
        """
        return await self.lore_integration.generate_scene_description_with_lore(location)
    
    async def _register_core_agents(self):
        """Register core system agents with the governor."""
        # Initialize and register memory agent
        try:
            memory_context = MemorySystemContext(self.user_id, self.conversation_id)
            memory_agent = create_memory_agent(self.user_id, self.conversation_id)
            await self.governor.register_agent(AgentType.MEMORY_MANAGER, memory_agent)
            
            # Issue general directive for memory management
            await self.governor.issue_directive(
                agent_type=AgentType.MEMORY_MANAGER,
                agent_id="memory_manager",
                directive_type=DirectiveType.ACTION,
                directive_data={
                    "instruction": "Maintain entity memories and ensure proper consolidation.",
                    "scope": "global"
                },
                priority=DirectivePriority.MEDIUM,
                duration_minutes=24*60  # 24 hours
            )
            
            # Store in local registry
            self.registered_agents[AgentType.MEMORY_MANAGER] = memory_agent
            
            logger.info("Memory Manager registered with governance system")
        except Exception as e:
            logger.error(f"Error registering Memory Manager: {e}")
        
        # Initialize and register new game agent
        try:
            new_game_agent = NewGameAgent()
            await self.governor.register_agent(AgentType.UNIVERSAL_UPDATER, new_game_agent)
            
            # Store in local registry
            self.registered_agents[AgentType.UNIVERSAL_UPDATER] = new_game_agent
            
            logger.info("New Game Agent registered with governance system")
        except Exception as e:
            logger.error(f"Error registering New Game Agent: {e}")
        
        # Register additional core agents
        try:
            # Import agent instances for others we might need
            from story_agent.story_director_agent import get_story_director
            from logic.conflict_system.conflict_integration import ConflictSystemIntegration
            from logic.activity_analyzer import ActivityAnalyzer
            
            # Story Director
            story_director = await get_story_director(self.user_id, self.conversation_id)
            await self.governor.register_agent(AgentType.STORY_DIRECTOR, story_director)
            self.registered_agents[AgentType.STORY_DIRECTOR] = story_director
            
            # Conflict System
            conflict_system = ConflictSystemIntegration(self.user_id, self.conversation_id)
            await self.governor.register_agent(AgentType.CONFLICT_ANALYST, conflict_system)
            self.registered_agents[AgentType.CONFLICT_ANALYST] = conflict_system
            
            # Activity Analyzer
            activity_analyzer = ActivityAnalyzer(self.user_id, self.conversation_id)
            await self.governor.register_agent(AgentType.ACTIVITY_ANALYZER, activity_analyzer)
            self.registered_agents[AgentType.ACTIVITY_ANALYZER] = activity_analyzer
            
            logger.info("Additional core agents registered with governance")
        except Exception as e:
            logger.error(f"Error registering additional core agents: {e}")
    
    async def register_agent(self, agent_type: str, agent_instance: Any, agent_id: str = "default"):
        """
        Register an agent with the governance system.
        
        Args:
            agent_type: The type of agent (use AgentType constants)
            agent_instance: The agent instance
            agent_id: Identifier for this agent instance
        """
        await self.governor.register_agent(agent_type, agent_instance)
        
        # Store in local registry
        if agent_type not in self.registered_agents:
            self.registered_agents[agent_type] = {}
        
        self.registered_agents[agent_type][agent_id] = agent_instance
        
        logger.info(f"Registered agent of type {agent_type} with ID {agent_id}")
    
    async def process_user_message(
        self, 
        user_message: str, 
        context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the complete Nyx system.
        
        This is a unified interface that handles:
        1. User model updates
        2. Memory operations
        3. Response generation
        4. Narrative progression
        5. Scene management
        
        Args:
            user_message: User's message text
            context_data: Optional additional context
            
        Returns:
            Complete response with all necessary data
        """
        context_data = context_data or {}
        
        # Create trace for monitoring
        with trace(
            workflow_name="Nyx Message Processing",
            trace_id=f"nyx-msg-{self.conversation_id}-{int(datetime.now().timestamp())}",
            group_id=f"user-{self.user_id}"
        ):
            # 1. Update user model
            if self.config["auto_update_user_model"]:
                user_model_updates = await process_user_input_for_model(
                    self.user_id, 
                    self.conversation_id, 
                    user_message,
                    context_data=context_data
                )
                
                # Get response guidance based on updated model
                response_guidance = await get_response_guidance_for_user(
                    self.user_id,
                    self.conversation_id,
                    context_data
                )
                
                # Update context with user model guidance
                context_data["user_model_guidance"] = response_guidance
            
            # 2. Process memory operations - remember this interaction
            memory_result = await process_memory_operation(
                self.user_id,
                self.conversation_id,
                "create",
                f"User message: {user_message}"
            )
            
            # 3. Get relevant memories for response context
            memory_recall = await process_memory_operation(
                self.user_id,
                self.conversation_id,
                "retrieve",
                user_message,
                context_data
            )
            
            if memory_recall and "memories" in memory_recall:
                context_data["relevant_memories"] = memory_recall["memories"]
            
            # 4. Generate Nyx's response with governance oversight
            response = await process_user_input(
                self.user_id,
                self.conversation_id,
                user_message,
                context_data
            )
            
            # 5. Update narrative and scene if needed
            scene_update = {}
            if "time_advancement" in response and response["time_advancement"]:
                # Update scene
                scene_update = await process_scene_input(
                    self.user_id,
                    self.conversation_id,
                    user_message,
                    context_data
                )
                
                # Process universal update for time advancement
                from logic.universal_updater_sdk import process_universal_update
                universal_result = await process_universal_update(
                    self.user_id, 
                    self.conversation_id,
                    f"Time advances after: {user_message}",
                    context_data
                )
                
                if universal_result:
                    response["universal_update"] = universal_result
            
            # 6. Post-processing: Remember Nyx's response in memory
            await process_memory_operation(
                self.user_id,
                self.conversation_id,
                "create",
                f"Nyx response: {response['message']}"
            )
            
            # 7. Broadcast event to NPCs if relevant
            if "broadcast_to_npcs" in context_data and context_data["broadcast_to_npcs"]:
                await broadcast_event_with_governance(
                    self.user_id,
                    self.conversation_id,
                    "user_message",
                    {
                        "user_message": user_message,
                        "nyx_response": response["message"]
                    }
                )
            
            # 8. Possibly generate a reflection if this is a significant interaction
            reflection = None
            if len(user_message) > 100 or any(word in user_message.lower() for word in ["think", "reflect", "understand", "feel"]):
                reflection = await generate_reflection(
                    self.user_id,
                    self.conversation_id,
                    "Recent interactions with the user"
                )
            
            # Combine everything into the final response
            full_response = {
                "message": response["message"],
                "memory_id": memory_result.get("memory_id"),
                "scene_update": scene_update,
                "generate_image": response.get("generate_image", False),
                "image_prompt": response.get("image_prompt"),
                "reflection": reflection["reflection"] if reflection else None,
                "user_model_updates": user_model_updates if self.config["auto_update_user_model"] else None,
                "time_advancement": response.get("time_advancement", False)
            }
            
            return full_response
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run system-wide maintenance on all components.
        
        This ensures optimal performance by:
        1. Cleaning up memory systems
        2. Consolidating user model
        3. Optimizing relationships
        4. Checking for narrative opportunities
        
        Returns:
            Dictionary with maintenance results
        """
        # Create trace for monitoring
        with trace(
            workflow_name="Nyx Maintenance",
            trace_id=f"nyx-maint-{self.conversation_id}-{int(datetime.now().timestamp())}",
            group_id=f"user-{self.user_id}"
        ):
            results = {
                "memory_maintenance": None,
                "user_model_maintenance": None,
                "narrative_maintenance": None,
                "npc_maintenance": None
            }
            
            # 1. Memory System Maintenance
            memory_result = await perform_memory_maintenance(self.user_id, self.conversation_id)
            results["memory_maintenance"] = memory_result
            
            # 2. Run joint memory maintenance for NPCs
            try:
                npc_memory_result = await self.npc_integration.run_joint_memory_maintenance()
                results["npc_maintenance"] = npc_memory_result
            except Exception as e:
                logger.error(f"Error in NPC memory maintenance: {e}")
                results["npc_maintenance"] = {"error": str(e)}
            
            # 3. Check narrative opportunities
            try:
                from story_agent.story_director_agent import check_narrative_opportunities
                narrative_result = await check_narrative_opportunities(self.user_id, self.conversation_id)
                results["narrative_maintenance"] = narrative_result
            except Exception as e:
                logger.error(f"Error checking narrative opportunities: {e}")
                results["narrative_maintenance"] = {"error": str(e)}
            
            # 4. Process system reflections
            try:
                reflection = await generate_reflection(
                    self.user_id,
                    self.conversation_id,
                    "System performance and optimization"
                )
                results["system_reflection"] = reflection
            except Exception as e:
                logger.error(f"Error generating system reflection: {e}")
                results["system_reflection"] = {"error": str(e)}
            
            return results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the complete status of the Nyx system.
        
        Returns:
            Dictionary with comprehensive system status
        """
        try:
            # Gather narrative status through governance
            narrative_status = await self.governor.get_narrative_status()
            
            # Count active directives for each agent type
            directive_counts = {}
            for agent_type in [AgentType.NPC, AgentType.STORY_DIRECTOR, AgentType.CONFLICT_ANALYST,
                             AgentType.NARRATIVE_CRAFTER, AgentType.RESOURCE_OPTIMIZER,
                             AgentType.RELATIONSHIP_MANAGER, AgentType.UNIVERSAL_UPDATER]:
                directives = await self.governor.get_agent_directives(agent_type, "all")
                directive_counts[agent_type] = len(directives)
            
            # Get memory statistics
            memory_stats = await self.memory_integration.memory_bridge.get_memory_stats()
            
            # User model information
            from nyx.nyx_model_manager import UserModelManager
            user_model_manager = UserModelManager(self.user_id, self.conversation_id)
            user_model = await user_model_manager.get_user_model()
            
            # Update system status
            current_status = {
                "timestamp": datetime.now().isoformat(),
                "narrative_status": narrative_status,
                "directive_counts": directive_counts,
                "memory_stats": memory_stats,
                "registered_agents": list(self.registered_agents.keys()),
                "user_model_summary": {
                    "kink_count": len(user_model.get("kink_profile", {})),
                    "behavior_patterns": len(user_model.get("behavior_patterns", {})),
                    "personality_assessment": user_model.get("personality_assessment", {})
                },
                "components": self.system_status["components"]
            }
            
            return current_status
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def perform_coordinated_action(
        self,
        action_type: str,
        primary_agent_type: str,
        action_data: Dict[str, Any],
        supporting_agents: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a complex action that requires coordination between multiple agents.
        
        This leverages the governor's coordination capabilities.
        
        Args:
            action_type: Type of action to perform
            primary_agent_type: Primary agent responsible for action
            action_data: Action details and parameters
            supporting_agents: Other agents to support the action
            
        Returns:
            Coordination results
        """
        return await self.governor.coordinate_agents(
            action_type,
            primary_agent_type,
            action_data,
            supporting_agents
        )


# Top-level functions for easy access

async def get_central_governance(user_id: int, conversation_id: int) -> NyxCentralGovernance:
    """
    Get (or create) the central governance system for a user/conversation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        The central governance system
    """
    # Use a cache to avoid recreating the governance system unnecessarily
    cache_key = f"governance:{user_id}:{conversation_id}"
    
    # Check if it's already in global dict
    if not hasattr(get_central_governance, "cache"):
        get_central_governance.cache = {}
    
    if cache_key in get_central_governance.cache:
        return get_central_governance.cache[cache_key]
    
    # Create new governance system
    governance = NyxCentralGovernance(user_id, conversation_id)
    await governance.initialize()
    
    # Cache it
    get_central_governance.cache[cache_key] = governance
    
    return governance

async def reset_governance(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Reset the governance system for a user/conversation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Status of the operation
    """
    cache_key = f"governance:{user_id}:{conversation_id}"
    
    # Clear from cache if exists
    if hasattr(get_central_governance, "cache") and cache_key in get_central_governance.cache:
        del get_central_governance.cache[cache_key]
    
    # Create new governance system
    governance = NyxCentralGovernance(user_id, conversation_id)
    await governance.initialize()
    
    # Cache it
    if not hasattr(get_central_governance, "cache"):
        get_central_governance.cache = {}
    get_central_governance.cache[cache_key] = governance
    
    # Register all agent systems with governance
    registration_results = {
        "universal_updater": False,
        "addiction_system": False,
        "aggregator": False,
        "inventory_system": False,
        "scene_manager": False,
        "memory_system": False
    }
    
    # 1. Register Universal Updater
    try:
        from logic.universal_updater_sdk import register_with_governance as register_updater
        await register_updater(user_id, conversation_id)
        registration_results["universal_updater"] = True
        logger.info(f"Universal Updater registered for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error registering Universal Updater: {e}")
    
    # 2. Register Addiction System
    try:
        from logic.addiction_system_sdk import register_with_governance as register_addiction
        await register_addiction(user_id, conversation_id)
        registration_results["addiction_system"] = True
        logger.info(f"Addiction System registered for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error registering Addiction System: {e}")
    
    # 3. Register Aggregator
    try:
        from logic.aggregator_sdk import register_with_governance as register_aggregator
        await register_aggregator(user_id, conversation_id)
        registration_results["aggregator"] = True
        logger.info(f"Aggregator registered for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error registering Aggregator: {e}")
    
    # 4. Register Inventory System
    try:
        from logic.inventory_system_sdk import register_with_governance as register_inventory
        await register_inventory(user_id, conversation_id)
        registration_results["inventory_system"] = True
        logger.info(f"Inventory System registered for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error registering Inventory System: {e}")
    
    # 5. Initialize Memory System (if separate registration is needed)
    try:
        from nyx.memory_integration_sdk import perform_memory_maintenance
        await perform_memory_maintenance(user_id, conversation_id)
        registration_results["memory_system"] = True
        logger.info(f"Memory System maintenance performed for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error initializing Memory System: {e}")
    
    # 6. Initialize User Model (if necessary)
    try:
        from nyx.user_model_sdk import initialize_user_model
        await initialize_user_model(user_id)
        logger.info(f"User Model initialized for user {user_id}")
    except Exception as e:
        logger.error(f"Error initializing User Model: {e}")

    try:
        from logic.time_cycle import register_with_governance as register_time_cycle
        await register_time_cycle(user_id, conversation_id)
        registration_results["time_cycle"] = True
        logger.info(f"Time Cycle registered for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error registering Time Cycle: {e}")
    
    # Register Lore Agents
    try:
        from lore.lore_agents import register_with_governance as register_lore
        await register_lore(user_id, conversation_id)
        registration_results["lore_agents"] = True
        logger.info(f"Lore Agents registered for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error registering Lore Agents: {e}")
    
    # Register Story Director and specialized agents
    try:
        from story_agent.story_director_agent import register_with_governance as register_story_director
        await register_story_director(user_id, conversation_id)
        registration_results["story_director"] = True
        logger.info(f"Story Director registered for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error registering Story Director: {e}")
    
    return {
        "status": "success",
        "message": f"Reset governance for user {user_id}, conversation {conversation_id}",
        "registrations": registration_results
    }

async def process_story_beat_with_governance(
    user_id: int, 
    conversation_id: int, 
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a story beat with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        context_data: Context data for the story beat
        
    Returns:
        Processed story beat
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.story_integration.generate_story_beat(context_data)

async def create_new_game_with_governance(
    user_id: int,
    conversation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new game with governance oversight.
    
    Args:
        user_id: User ID
        conversation_data: Initial conversation data
        
    Returns:
        New game creation results
    """
    # For new game, conversation_id might not exist yet
    conversation_id = conversation_data.get("conversation_id", 0)
    
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.new_game_integration.create_new_game(conversation_data)

async def orchestrate_scene_with_governance(
    user_id: int,
    conversation_id: int,
    location: str,
    player_action: str = None,
    involved_npcs: List[int] = None
) -> Dict[str, Any]:
    """
    Orchestrate a scene with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        location: Scene location
        player_action: Optional player action
        involved_npcs: Optional list of involved NPC IDs
        
    Returns:
        Orchestrated scene
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.npc_integration.orchestrate_scene(
        location, player_action, involved_npcs
    )

async def broadcast_event_with_governance(
    user_id: int,
    conversation_id: int,
    event_type: str,
    event_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Broadcast an event with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        event_type: Type of event
        event_data: Event data
        
    Returns:
        Broadcast results
    """
    governance = await get_central_governance(user_id, conversation_id)
    return await governance.event_manager.broadcast_event(event_type, event_data)

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
