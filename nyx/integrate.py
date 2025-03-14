# nyx/integrate.py

"""
Enhanced integration module for Nyx's central governance system.

This module extends nyx/integrate.py to implement the unified governance functionality
from nyx_governance.py, ensuring Nyx properly oversees and controls all aspects of the game,
including story generation, new game creation, and NPC management.
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

# Import original integration components we'll enhance
from nyx.integrate import JointMemoryGraph, GameEventManager, NyxNPCIntegrationManager

# Import story components
from story_agent.agent_interaction import (
    orchestrate_conflict_analysis_and_narrative,
    generate_comprehensive_story_beat
)
from story_agent.story_director_agent import initialize_story_director

# Import new game components
from new_game_agent import NewGameAgent

# The agent trace utility
from agents import trace

# Database connection helper
from db.connection import get_db_connection

# Caching utilities
from utils.caching import CACHE_TTL, NPC_DIRECTIVE_CACHE, AGENT_DIRECTIVE_CACHE

logger = logging.getLogger(__name__)


class NyxCentralGovernance:
    """
    Enhanced central governance system that integrates NyxUnifiedGovernor with all aspects 
    of the application, providing centralized control over story, new game creation, and NPCs.
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
        self.memory_graph = EnhancedJointMemoryGraph(user_id, conversation_id, self.governor)
        self.event_manager = EnhancedGameEventManager(user_id, conversation_id, self.governor)
        self.npc_integration = EnhancedNPCIntegrationManager(user_id, conversation_id, self.governor)
        
        # Components for specialized areas
        self.story_integration = StoryIntegration(user_id, conversation_id, self.governor)
        self.new_game_integration = NewGameIntegration(user_id, conversation_id, self.governor)
        self.memory_integration = MemoryIntegration(user_id, conversation_id, self.governor)  # Add this line
        
        # Used to track registered agents
        self.registered_agents = {}

    async def initialize(self):
        """Initialize the governance system and ensure database tables exist."""
        await self.governor.setup_database_tables()
        
        # Register core system agents
        await self._register_core_agents()
        
        # Initialize memory integration
        await self.memory_integration.initialize()
        
        logger.info(f"NyxCentralGovernance initialized for user {self.user_id}, conversation {self.conversation_id}")
    
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
            
            logger.info("Memory Manager registered with governance system")
        except Exception as e:
            logger.error(f"Error registering Memory Manager: {e}")
        
        # Initialize and register new game agent
        try:
            new_game_agent = NewGameAgent()
            await self.governor.register_agent(AgentType.UNIVERSAL_UPDATER, new_game_agent)
            logger.info("New Game Agent registered with governance system")
        except Exception as e:
            logger.error(f"Error registering New Game Agent: {e}")
    
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
    
    async def check_action_permission(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Check if an agent is allowed to perform an action.
        
        Args:
            agent_type: Type of agent (NPC, STORY_DIRECTOR, etc.)
            agent_id: ID of the agent
            action_type: Type of action
            action_details: Details of the action
            context: Additional context
            
        Returns:
            Permission check results
        """
        return await self.governor.check_action_permission(
            agent_type, agent_id, action_type, action_details, context
        )
    
    async def issue_directive(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        directive_type: str,
        directive_data: Dict[str, Any],
        priority: int = DirectivePriority.MEDIUM,
        duration_minutes: int = 30,
        scene_id: str = None
    ) -> int:
        """
        Issue a directive to an agent.
        
        Args:
            agent_type: Type of agent
            agent_id: ID of the agent
            directive_type: Type of directive
            directive_data: Directive data
            priority: Directive priority
            duration_minutes: Duration in minutes
            scene_id: Optional scene ID
            
        Returns:
            Directive ID
        """
        return await self.governor.issue_directive(
            agent_type, agent_id, directive_type, directive_data,
            priority, duration_minutes, scene_id
        )
    
    async def process_agent_action_report(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process an action report from an agent.
        
        Args:
            agent_type: Type of agent
            agent_id: ID of the agent
            action: Action performed
            result: Result of the action
            context: Additional context
            
        Returns:
            Processing results
        """
        return await self.governor.process_agent_action_report(
            agent_type, agent_id, action, result, context
        )
    
    async def coordinate_agents(
        self,
        action_type: str,
        primary_agent_type: str,
        action_details: Dict[str, Any],
        supporting_agents: List[str] = None
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for a complex action.
        
        Args:
            action_type: Type of action
            primary_agent_type: Primary agent type
            action_details: Action details
            supporting_agents: Supporting agent types
            
        Returns:
            Coordination results
        """
        return await self.governor.coordinate_agents(
            action_type, primary_agent_type, action_details, supporting_agents
        )
    
    async def broadcast_to_all_agents(self, message_type: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcast a message to all registered agents.
        
        Args:
            message_type: Type of message
            message_data: Message data
            
        Returns:
            Broadcast results
        """
        return await self.governor.broadcast_to_all_agents(message_type, message_data)
    
    async def get_narrative_status(self) -> Dict[str, Any]:
        """
        Get the current status of the narrative.
        
        Returns:
            Dictionary with narrative status
        """
        return await self.governor.get_narrative_status()
    
    async def run_joint_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run joint memory maintenance for Nyx and NPCs.
        
        Returns:
            Maintenance results
        """
        return await self.npc_integration.run_joint_memory_maintenance(
            self.user_id, self.conversation_id
        )


class EnhancedJointMemoryGraph(JointMemoryGraph):
    """Enhanced Joint Memory Graph with governance integration."""
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor):
        super().__init__(user_id, conversation_id)
        self.governor = governor
    
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
        
        # Check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=source_type,
            agent_id=source_id,
            action_type="add_memory",
            action_details={
                "memory_text": memory_text,
                "significance": significance,
                "shared_with": shared_with
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Memory creation not approved: {permission['reasoning']}")
            return -1
        
        # Proceed with original implementation
        memory_id = await super().add_joint_memory(
            memory_text, source_type, source_id, shared_with, 
            significance, tags, metadata
        )
        
        # Report the action
        await self.governor.process_agent_action_report(
            agent_type=source_type,
            agent_id=source_id,
            action={
                "type": "add_memory",
                "description": f"Added joint memory: {memory_text[:50]}..."
            },
            result={
                "memory_id": memory_id,
                "significance": significance
            }
        )
        
        return memory_id

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


class EnhancedGameEventManager(GameEventManager):
    """Enhanced Game Event Manager with governance integration."""
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor
        
        # We don't call super().__init__() because the original implementation 
        # has hard-coded attribute assignments that don't match our needs
        
        # Initialize required components
        self.nyx_agent_sdk = self.get_nyx_agent(user_id, conversation_id)
        self.npc_coordinator = self.get_npc_coordinator(user_id, conversation_id)
    
    async def broadcast_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced broadcast event with governance oversight.
        """
        logger.info(f"Broadcasting event {event_type} with governance oversight")
        
        # Check permission with governance system first
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
        
        # If approved, proceed with enhanced broadcast logic
        
        # Tell Nyx about the event FIRST and get filtering instructions
        nyx_response = await self.nyx_agent_sdk.process_game_event(event_type, event_data)
        
        # Check if Nyx wants to filter or modify this event
        if not nyx_response.get("should_broadcast_to_npcs", True):
            logger.info(f"Nyx has blocked broadcasting event {event_type} to NPCs: {nyx_response.get('reason', 'No reason provided')}")
            
            # Report the action
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
            
            # Report modification
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
            
            # Issue directives to affected NPCs if needed
            if event_type in ["conflict_update", "critical_event", "emergency"]:
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


class EnhancedNPCIntegrationManager(NyxNPCIntegrationManager):
    """Enhanced NPC Integration Manager with governance integration."""
    
    def __init__(self, user_id: int, conversation_id: int, governor: NyxUnifiedGovernor):
        """Initialize with governor access."""
        # Initialize with original method
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = governor
        
        # Initialize other components - simulate super().__init__ behavior
        # since we can't directly call it with the extra governor parameter
        self.nyx_agent_sdk = self.get_nyx_agent(user_id, conversation_id)
        self.npc_coordinator = self.get_npc_coordinator(user_id, conversation_id)
        self.npc_system = None  # Lazy-loaded
    
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
        
        # Issue directives to NPCs if specified
        if involved_npcs:
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
        
        # Report the action
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
        
        # If approved by governor, proceed with original implementation
        original_result = await super().approve_group_interaction(request)
        
        # Add governance flag
        original_result["governance_approved"] = True
        
        return original_result


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
