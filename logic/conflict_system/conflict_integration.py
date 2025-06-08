# logic/conflict_system/conflict_integration.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from agents import Agent, Runner, trace, function_tool, RunContextWrapper
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance
from lore.lore_system import LoreSystem
from db.connection import get_db_connection_context
from lore.core import canon
from logic.fully_integrated_npc_system import IntegratedNPCSystem
from logic.conflict_system.conflict_agents import (
    triage_agent, conflict_generation_agent, stakeholder_agent,
    manipulation_agent, resolution_agent, ConflictContext, initialize_agents
)
from npcs.npc_agent_system import NPCAgentSystem
from logic.conflict_system.conflict_tools import (
    get_active_conflicts, get_conflict_details, get_conflict_stakeholders,
    get_resolution_paths, get_player_involvement, get_internal_conflicts,
    update_conflict_progress, update_stakeholder_status, add_resolution_path,
    update_player_involvement, add_internal_conflict, resolve_internal_conflict,
    generate_conflict_consequences
)

# Import story agent components
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager
from context.vector_service import get_vector_service
from story_agent.story_director_agent import (
    initialize_story_director, get_current_story_state
)

logger = logging.getLogger(__name__)

class ConflictSystemIntegration:
    """
    Combined integration of conflict system and Nyx governance, with enhanced
    context-aware integration with the agentic story architecture.
    """
    _npc_systems: Dict[str, IntegratedNPCSystem] = {}

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_id = "conflict_manager"
        self.is_initialized = False
        self.agents = None
        self.lore_system = None
        self.npc_system = None
        self.story_context = None
        self.run_ctx = RunContextWrapper({
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
        })
        
        # Context and agent-related systems
        self.context_service = None
        self.memory_manager = None
        self.vector_service = None
        self.story_director = None
        self.story_director_context = None

        # Directive handlers
        self.directive_handlers = {
            DirectiveType.ACTION: self._handle_action_directive,
            DirectiveType.SCENE: self._handle_scene_directive,
            DirectiveType.PROHIBITION: self._handle_prohibition_directive,
            DirectiveType.INFORMATION: self._handle_information_directive
        }

    # -------------------------- INITIALIZATION -------------------------- #

    async def initialize(self):
        if not self.is_initialized:
            logger.info(f"Initializing conflict system for user {self.user_id}")
            
            # Start a trace for the initialization process
            with trace(workflow_name="ConflictSystemInit", group_id=f"conflict_{self.conversation_id}"):
                # Initialize core components
                self.agents = await initialize_agents()
                self.lore_system = await self.get_lore_system(self.user_id, self.conversation_id)
                self.npc_system = await self.get_npc_system(self.user_id, self.conversation_id)
                
                # Initialize player stats if needed
                from logic.narrative_progression import initialize_player_stats
                await initialize_player_stats(self.user_id, self.conversation_id)
                
                # Initialize context-related systems properly
                self.context_service = await get_context_service(self.user_id, self.conversation_id)
                self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
                self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
                
                # Initialize story director with proper context handling
                self.story_director, self.story_director_context = await initialize_story_director(
                    self.user_id, self.conversation_id
                )
                
                self.story_context = await self._get_story_context()
                self.is_initialized = True
                
        return self

    @staticmethod
    async def get_lore_system(user_id: int, conversation_id: int):
        lore_system = LoreSystem.get_instance(user_id, conversation_id)
        await lore_system.initialize()
        return lore_system

    @classmethod
    async def get_npc_system(cls, user_id: int, conversation_id: int):
        key = f"{user_id}:{conversation_id}"
        if key not in cls._npc_systems:
            cls._npc_systems[key] = IntegratedNPCSystem(user_id, conversation_id)
            await cls._npc_systems[key].initialize()
            logger.info(f"Created new IntegratedNPCSystem for user={user_id}, conversation={conversation_id}")
        return cls._npc_systems[key]

    # -------------------------- STORY CONTEXT / DATA -------------------------- #

    async def _get_story_context(self) -> Dict[str, Any]:
        """
        Gather everything needed to resume or display the current story:
          - active NPCs
          - lore elements for the current narrative
          - story progression details
          - the narrative ID itself
        
        Uses context service and story director for enhanced integration.
        """
        default = {
            "active_npcs": [],
            "lore_context": [],
            "story_progression": {},
            "current_narrative_id": None,
        }
        
        try:
            # 1) Get comprehensive context from context service
            comprehensive_context = await self.context_service.get_context()
            
            # 2) Fetch active NPCs from context
            active_npcs = comprehensive_context.get("introduced_npcs", [])
            if not active_npcs:
                # Fallback to NPC system if needed
                active_npcs = await self.npc_system._get_active_npcs()
        
            # 3) Current narrative ID & lore
            narrative_id = await self._get_current_narrative_id()
            lore_context = (
                await self.lore_system.get_narrative_elements(narrative_id)
                if narrative_id else []
            )
        
            # 4) Story progression from story director or context
            story_progression = await self._get_story_progression()
        
            return {
                "active_npcs": active_npcs,
                "lore_context": lore_context,
                "story_progression": story_progression,
                "current_narrative_id": narrative_id,
                "comprehensive_context": comprehensive_context
            }
        
        except Exception as e:
            logger.error(f"Error getting story context: {e}")
            return default

    async def _get_current_narrative_id(self) -> Optional[int]:
        try:
            # Try context service first
            if self.context_service:
                context = await self.context_service.get_context()
                if "current_narrative" in context:
                    return context["current_narrative"].get("id")
            
            # Fallback to database
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    "SELECT value FROM CurrentRoleplay WHERE key = 'CurrentNarrativeId' AND user_id = $1 AND conversation_id = $2",
                    self.user_id, self.conversation_id
                )
                return int(row['value']) if row and row['value'] else None
        except Exception as e:
            logger.error(f"Error getting current narrative ID: {e}")
            return None

    async def _get_story_progression(self) -> Dict[str, Any]:
        """
        Get story progression information using the context service and story director.
        This replaces the direct database query to the missing StoryProgression table.
        """
        try:
            # Method 1: Try to get from story director's current state
            if self.story_director and self.story_director_context:
                try:
                    # FIXED: Pass the context directly without trying to access a 'context' attribute
                    # The story_director_context IS the context object, not a container for it
                    story_state = await get_current_story_state(self.story_director, self.story_director_context)
                    
                    if story_state and hasattr(story_state, 'final_output'):
                        state_text = story_state.final_output
                        
                        # Try to extract JSON data from the response if present
                        try:
                            import re
                            json_match = re.search(r'({.*})', state_text)
                            if json_match:
                                json_str = json_match.group(1)
                                state_data = json.loads(json_str)
                                
                                # Extract relevant progression data
                                narrative_stage = state_data.get("narrative_stage", {})
                                return {
                                    'current_day': state_data.get("current_day", 1),
                                    'progress': narrative_stage.get("progress", 0),
                                    'phase': narrative_stage.get("name", "beginning"),
                                    'key_decisions': state_data.get("key_decisions", [])
                                }
                        except json.JSONDecodeError:
                            pass
                except Exception as e:
                    logger.warning(f"Error getting story state from director: {e}")
            
            # Method 2: Try context service
            if self.context_service:
                context_data = await self.context_service.get_context()
                
                # Extract time info for current day
                time_info = context_data.get("time_info", {})
                current_day = int(time_info.get("day", 1)) if time_info.get("day", "").isdigit() else 1
                
                # Extract narrative stage info
                narrative_stage = context_data.get("narrative_stage", {})
                
                return {
                    'current_day': current_day,
                    'progress': narrative_stage.get("progress", 0),
                    'phase': narrative_stage.get("name", "beginning"),
                    'key_decisions': narrative_stage.get("key_decisions", [])
                }
            
            # Method 3: Fall back to database for basic day info
            async with get_db_connection_context() as conn:
                day_row = await conn.fetchrow(
                    "SELECT value FROM CurrentRoleplay WHERE key = 'CurrentDay' AND user_id = $1 AND conversation_id = $2",
                    self.user_id, self.conversation_id
                )
                current_day = int(day_row['value']) if day_row and day_row['value'] else 1
                
                # Try to get any progress information from related tables
                # (This is a fallback and can be customized based on your schema)
                progress = 0
                phase = "beginning"
                
                return {
                    'current_day': current_day,
                    'progress': progress,
                    'phase': phase,
                    'key_decisions': []
                }
        except Exception as e:
            logger.error(f"Error getting story progression: {e}")
            return {'current_day': 1}

    # ------------------ DIRECTIVE HANDLING ------------------ #

    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Universal directive handler—calls the appropriate directive type handler.
        """
        await self.initialize()
        directive_type = directive.get("type")
        directive_data = directive.get("data", {})
        logger.info(f"Conflict system received directive: {directive_type}")
        logger.debug(f"Directive data: {json.dumps(directive_data, indent=2)}")
        handler = self.directive_handlers.get(directive_type)
        if handler:
            return await handler(directive_data)
        else:
            return {"success": False, "message": f"Unhandled directive type: {directive_type}", "directive_type": directive_type}

    async def _handle_action_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handles action requests from governance."""
        action_type = directive_data.get("action_type")
        action_params = directive_data.get("parameters", {})
        augmented_params = {**action_params, "enhanced": True}
        
        if action_type == "generate_conflict":
            return await self.generate_conflict(augmented_params)
        elif action_type == "resolve_conflict":
            return await self.resolve_conflict(augmented_params)
        elif action_type == "update_stakeholders":
            return await self.update_stakeholders(augmented_params)
        elif action_type == "manage_manipulation":
            return await self.manage_manipulation(augmented_params)
        else:
            return {"success": False, "message": f"Unknown action type: {action_type}"}

    async def _handle_scene_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        scene_type = directive_data.get("scene_type")
        location = directive_data.get("location")
        participants = directive_data.get("participants", [])
        logger.info(f"Scene change to {location}, type: {scene_type}, participants: {len(participants)}")

        # Auto background conflict for certain scenes
        if scene_type == "public_space" and len(participants) > 3:
            conflict_data = {
                "conflict_type": "background",
                "location": location,
                "intensity": "low",
                "player_involvement": "observable",
                "duration": "short",
                "existing_npcs": participants
            }
            asyncio.create_task(self.generate_conflict(conflict_data))
        
        # Create a memory of this scene change
        if self.memory_manager:
            await self.memory_manager.add_memory(
                content=f"Scene changed to {location}, type: {scene_type}",
                memory_type="scene_change",
                importance=0.6,
                tags=["conflict", "scene_change", location.lower().replace(" ", "_")],
                metadata={
                    "scene_type": scene_type,
                    "location": location,
                    "participants": participants
                }
            )
            
        return {"success": True, "message": "Conflict system is aware of scene change", "scene_type": scene_type, "location": location}

    async def _handle_prohibition_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        prohibited_action = directive_data.get("prohibited_action")
        reason = directive_data.get("reason")
        duration = directive_data.get("duration_minutes", 60)
        logger.info(f"Registering prohibition: {prohibited_action} for reason: {reason}, duration: {duration}")
        
        # Record the prohibition in the memory system
        if self.memory_manager:
            await self.memory_manager.add_memory(
                content=f"Prohibited action: {prohibited_action} - {reason}",
                memory_type="prohibition",
                importance=0.7,
                tags=["conflict", "prohibition"],
                metadata={
                    "prohibited_action": prohibited_action,
                    "reason": reason,
                    "duration_minutes": duration
                }
            )
            
        return {"success": True, "message": f"Prohibition registered: {prohibited_action}", "duration_minutes": duration}

    async def _handle_information_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        info_type = directive_data.get("info_type")
        info_content = directive_data.get("content")
        logger.info(f"Processing information: type={info_type}")
        
        # Record the information in the memory system
        if self.memory_manager:
            await self.memory_manager.add_memory(
                content=f"Information received: {info_type} - {info_content}",
                memory_type="information",
                importance=0.5,
                tags=["conflict", "information", info_type],
                metadata={
                    "info_type": info_type,
                    "content": info_content
                }
            )
            
        return {"success": True, "message": f"Information processed: {info_type}", "info_type": info_type}

    # ------------------- CONFLICT MANAGEMENT AND RESOLUTION ------------------- #

    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="generate_conflict",
        action_description="Generate a new conflict with stakeholders"
    )
    async def generate_conflict(
        self, 
        conflict_data_or_type: Union[Dict[str, Any], str, None] = None
    ) -> Dict[str, Any]:
        """
        Generate a new conflict with stakeholders and resolution paths.
        
        Args:
            conflict_data_or_type: Either a conflict type string (minor, standard, major, catastrophic)
                                   or a complete conflict data dictionary
            
        Returns:
            The generated conflict details
        """
        await self.initialize()
        try:
            # Handle string (conflict_type) or None input
            if conflict_data_or_type is None or isinstance(conflict_data_or_type, str):
                conflict_type = conflict_data_or_type
                
                # Create context for using conflict tools
                ctx = RunContextWrapper({"user_id": self.user_id, "conversation_id": self.conversation_id})
                
                # Get current day for conflict creation
                from logic.conflict_system.conflict_tools import get_current_day, get_available_npcs, generate_conflict_details
                current_day = await get_current_day(ctx)
                
                # Get active conflicts to determine appropriate type if not specified
                active_conflicts = await self.get_active_conflicts()
                
                # If there are already too many active conflicts (3+), make this a minor one
                if len(active_conflicts) >= 3 and not conflict_type:
                    conflict_type = "minor"
                
                # If there are no conflicts at all and no type specified, make this a standard one
                if len(active_conflicts) == 0 and not conflict_type:
                    conflict_type = "standard"
                
                # If still no type specified, choose randomly with weighted probabilities
                if not conflict_type:
                    weights = {
                        "minor": 0.4,
                        "standard": 0.4,
                        "major": 0.15,
                        "catastrophic": 0.05
                    }
                    
                    import random
                    conflict_type = random.choices(
                        list(weights.keys()),
                        weights=list(weights.values()),
                        k=1
                    )[0]
                
                # Get available NPCs to use as potential stakeholders
                npcs = await get_available_npcs(ctx)
                
                if len(npcs) < 3:
                    return {"success": False, "message": "Not enough NPCs available to create a complex conflict"}
                
                # Determine how many stakeholders to involve
                stakeholder_count = {
                    "minor": min(3, len(npcs)),
                    "standard": min(4, len(npcs)),
                    "major": min(5, len(npcs)),
                    "catastrophic": min(6, len(npcs))
                }.get(conflict_type, min(4, len(npcs)))
                
                # Select NPCs to involve as stakeholders
                stakeholder_npcs = random.sample(npcs, stakeholder_count)
                
                # Generate conflict details using AI
                conflict_data = await generate_conflict_details(ctx, conflict_type, stakeholder_npcs, current_day)
            else:
                # Use the provided conflict_data dictionary
                conflict_data = conflict_data_or_type
            
            # Ensure conflict_data is valid before continuing
            if not conflict_data or not isinstance(conflict_data, dict):
                raise ValueError(f"Invalid conflict data: {conflict_data}")
                
            # Get updated story context
            story_context = await self._get_story_context()
            
            enhanced_data = dict(conflict_data)
            enhanced_data.update({
                "story_context": story_context,
                "active_npcs": story_context.get("active_npcs"),
                "lore_context": story_context.get("lore_context")
            })
    
            conflict_context = ConflictContext(self.user_id, self.conversation_id)
            generation_result = await Runner.run(
                self.agents["conflict_generation_agent"],
                json.dumps(enhanced_data),
                context=conflict_context,
            )
            result = getattr(generation_result, 'final_output', generation_result)
    
            # Stakeholders
            stakeholder_data = {
                "conflict_id": result.get("conflict_id", getattr(result, 'conflict_id', None)),
                "conflict_details": result.get("conflict_details", {}),
                "existing_npcs": enhanced_data.get("existing_npcs", []),
                "required_roles": enhanced_data.get("required_roles", []),
                "story_context": story_context
            }
            stakeholder_result = await Runner.run(
                self.agents["stakeholder_agent"],
                json.dumps(stakeholder_data), context=conflict_context
            )
    
            # Lore event update
            await self.lore_system.handle_narrative_event(
                self.run_ctx,
                f"New conflict generated: {result.get('conflict_name', '')}",
                affected_lore_ids=result.get('affected_lore_ids', []),
                resolution_type="conflict_generation",
                impact_level="medium"
            )
            
            # Create a memory for this conflict
            if self.memory_manager:
                await self.memory_manager.add_memory(
                    content=f"New conflict generated: {result.get('conflict_name', '')}",
                    memory_type="conflict_generation",
                    importance=0.8,
                    tags=["conflict", "generation"],
                    metadata={
                        "conflict_id": result.get("conflict_id"),
                        "conflict_name": result.get("conflict_name", ""),
                        "conflict_type": result.get("conflict_type", ""),
                        "phase": "brewing",
                        "location": enhanced_data.get("location", "Unknown")
                    }
                )
    
            return {
                "success": True,
                "conflict_id": result.get("conflict_id"),
                "conflict_details": result.get("conflict_details"),
                "stakeholders": stakeholder_result.final_output.get("stakeholders", []),
                "player_involvement": result.get("player_involvement"),
                "story_context": story_context
            }
        except Exception as e:
            logger.error(f"Error generating conflict: {str(e)}", exc_info=True)
            return {"success": False, "message": str(e)}

    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="resolve_conflict",
        action_description="Resolve an existing conflict"
    )
    async def resolve_conflict(self, resolution_data: Dict[str, Any]) -> Dict[str, Any]:
        await self.initialize()
        try:
            story_context = await self._get_story_context()
            enhanced_data = dict(resolution_data)
            enhanced_data.update({
                "story_context": story_context,
                "active_npcs": story_context.get("active_npcs"),
                "lore_context": story_context.get("lore_context")
            })

            conflict_context = ConflictContext(self.user_id, self.conversation_id)
            resolution_result = await Runner.run(
                self.agents["resolution_agent"],
                json.dumps(enhanced_data),
                context=conflict_context,
            )
            result = getattr(resolution_result, 'final_output', resolution_result)

            await self.lore_system.handle_narrative_event(
                self.run_ctx,
                f"Conflict resolved: {result.get('description', '')}",
                affected_lore_ids=result.get('affected_lore_ids', []),
                resolution_type=enhanced_data.get("resolution_type", "standard"),
                impact_level="medium"
            )
            
            # Create a memory for this conflict resolution
            if self.memory_manager:
                await self.memory_manager.add_memory(
                    content=f"Conflict resolved: {result.get('description', '')}",
                    memory_type="conflict_resolution",
                    importance=0.8, 
                    tags=["conflict", "resolution"],
                    metadata={
                        "conflict_id": enhanced_data.get("conflict_id"),
                        "resolution_type": enhanced_data.get("resolution_type", "standard"),
                        "impact_level": "medium"
                    }
                )

            return {
                "success": True,
                "conflict_id": enhanced_data.get("conflict_id"),
                "resolution": result,
                "story_context": story_context
            }
        except Exception as e:
            logger.error(f"Error resolving conflict: {str(e)}", exc_info=True)
            return {"success": False, "message": str(e)}

    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="update_stakeholders",
        action_description="Update conflict stakeholders"
    )
    async def update_stakeholders(self, stakeholder_data: Dict[str, Any]) -> Dict[str, Any]:
        await self.initialize()
        try:
            conflict_context = ConflictContext(self.user_id, self.conversation_id)
            stakeholder_result = await Runner.run(
                self.agents["stakeholder_agent"], 
                json.dumps(stakeholder_data),
                context=conflict_context
            )
            result = getattr(stakeholder_result, 'final_output', stakeholder_result)
            
            # Create a memory about stakeholder updates
            if self.memory_manager:
                stakeholder_changes = []
                if result.get("added_stakeholders"):
                    stakeholder_changes.append(f"Added: {', '.join([s.get('npc_name', 'Unknown') for s in result.get('added_stakeholders', [])])}")
                if result.get("removed_stakeholders"):
                    stakeholder_changes.append(f"Removed: {', '.join([s.get('npc_name', 'Unknown') for s in result.get('removed_stakeholders', [])])}")
                if result.get("updated_stakeholders"):
                    stakeholder_changes.append(f"Updated: {', '.join([s.get('npc_name', 'Unknown') for s in result.get('updated_stakeholders', [])])}")
                
                changes_text = "; ".join(stakeholder_changes)
                await self.memory_manager.add_memory(
                    content=f"Updated stakeholders for conflict ID {stakeholder_data.get('conflict_id')}: {changes_text}",
                    memory_type="stakeholder_update",
                    importance=0.6,
                    tags=["conflict", "stakeholders"],
                    metadata={
                        "conflict_id": stakeholder_data.get("conflict_id"),
                        "changes": stakeholder_changes
                    }
                )
            
            return {
                "success": True,
                "conflict_id": stakeholder_data.get("conflict_id"),
                "updated_stakeholders": result.get("updated_stakeholders", []),
                "added_stakeholders": result.get("added_stakeholders", []),
                "removed_stakeholders": result.get("removed_stakeholders", [])
            }
        except Exception as e:
            logger.error(f"Error updating conflict stakeholders: {str(e)}", exc_info=True)
            return {"success": False, "message": str(e)}

    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="manage_manipulation",
        action_description="Manage NPC manipulation attempts"
    )
    async def manage_manipulation(self, manipulation_data: Dict[str, Any]) -> Dict[str, Any]:
        await self.initialize()
        try:
            conflict_context = ConflictContext(self.user_id, self.conversation_id)
            manipulation_result = await Runner.run(
                self.agents["manipulation_agent"],
                json.dumps(manipulation_data),
                context=conflict_context
            )
            result = getattr(manipulation_result, 'final_output', manipulation_result)
            
            # Create a memory about this manipulation
            if self.memory_manager:
                await self.memory_manager.add_memory(
                    content=f"Manipulation attempt: {result.get('description', 'No description')}",
                    memory_type="manipulation",
                    importance=0.7,
                    tags=["conflict", "manipulation"],
                    metadata={
                        "manipulation_type": result.get("type", "unknown"),
                        "success": result.get("success", False),
                        "target": result.get("target", "unknown")
                    }
                )
            
            return {
                "success": True,
                "manipulation_result": result
            }
        except Exception as e:
            logger.error(f"Error managing manipulation: {str(e)}", exc_info=True)
            return {"success": False, "message": str(e)}

    # -------------------------- HELPER METHODS -------------------------- #
    
    async def get_conflicts_with_context(self, query: str = None) -> List[Dict[str, Any]]:
        """
        Get conflicts using vector search for relevance.
        
        Args:
            query: Optional query for finding relevant conflicts
            
        Returns:
            List of conflict data
        """
        try:
            # Use vector search if available
            if self.vector_service and query:
                vector_results = await self.vector_service.search_entities(
                    query_text=query,
                    entity_types=["conflict"],
                    top_k=5
                )
                
                conflicts = []
                for result in vector_results:
                    if "metadata" in result and "conflict_id" in result["metadata"]:
                        conflict_id = result["metadata"]["conflict_id"]
                        conflict_details = await get_conflict_details(
                            RunContextWrapper({'user_id': self.user_id, 'conversation_id': self.conversation_id}), 
                            conflict_id
                        )
                        if conflict_details:
                            conflicts.append(conflict_details)
                
                return conflicts
            
            # Fallback to getting active conflicts
            active_conflicts = await get_active_conflicts(
                RunContextWrapper({'user_id': self.user_id, 'conversation_id': self.conversation_id})
            )
            
            return active_conflicts
        except Exception as e:
            logger.error(f"Error getting conflicts with context: {e}")
            return []

    async def _apply_conflict_rewards(self, consequences: List[Dict[str, Any]]) -> None:
        """
        Apply rewards/consequences from conflict resolution with memory integration.
        """
        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    canon_ctx = type("ctx", (), {"user_id": self.user_id, "conversation_id": self.conversation_id})()
                    for consequence in consequences:
                        ctype = consequence.get("type")
                        # Player stat changes
                        if ctype == "player_stat" and "stat_changes" in consequence:
                            for stat, change in consequence["stat_changes"].items():
                                current_val = await conn.fetchval(
                                    f"SELECT {stat} FROM PlayerStats WHERE user_id=$1 AND conversation_id=$2",
                                    self.user_id,
                                    self.conversation_id,
                                )
                                new_val = (current_val or 0) + change
                                await canon.update_player_stat_canonically(
                                    canon_ctx,
                                    conn,
                                    "Chase",
                                    stat,
                                    new_val,
                                    reason="conflict_reward",
                                )
                                logger.info(f"Updated player stat {stat} by {change} for user {self.user_id}")
                                
                                # Record this change in memory
                                if self.memory_manager:
                                    await self.memory_manager.add_memory(
                                        content=f"Player stat {stat} changed by {change} from conflict resolution",
                                        memory_type="stat_change",
                                        importance=0.6,
                                        tags=["conflict", "reward", "stat_change"],
                                        metadata={
                                            "stat": stat,
                                            "change": change,
                                            "source": "conflict_resolution"
                                        }
                                    )
                                
                        # Item rewards
                        elif ctype == "item_reward" and "item" in consequence:
                            item = consequence["item"]
                            await canon.find_or_create_inventory_item(
                                canon_ctx,
                                conn,
                                item_name=item["name"],
                                player_name="Chase",
                                item_description=item["description"],
                                item_category=item.get("category", "conflict_reward"),
                                item_properties={
                                    "rarity": item.get("rarity", "common"),
                                    "resolution_style": item.get("resolution_style", "neutral"),
                                    "source": "conflict_resolution",
                                },
                                quantity=1,
                                equipped=False,
                            )
                            logger.info(f"Added item {item['name']} to inventory for user {self.user_id}")
                            
                            # Record this reward in memory
                            if self.memory_manager:
                                await self.memory_manager.add_memory(
                                    content=f"Received item reward: {item['name']} from conflict resolution",
                                    memory_type="item_reward",
                                    importance=0.7,
                                    tags=["conflict", "reward", "item"],
                                    metadata={
                                        "item_name": item["name"],
                                        "item_category": item.get("category", "conflict_reward"),
                                        "source": "conflict_resolution"
                                    }
                                )
                                
                        # Other reward types follow the same pattern...
                        # [The rest of the reward code remains the same, just add memory creation where appropriate]
                            
        except Exception as e:
            logger.error(f"Error applying conflict rewards: {str(e)}", exc_info=True)
            raise

    @classmethod
    async def register_enhanced_integration(cls, user_id: int, conversation_id: int):
        """
        Register this conflict system integration with the central governance system.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
            
        Returns:
            The registered ConflictSystemIntegration instance
        """
        try:
            logger.info(f"Registering conflict system integration for user={user_id}, conversation={conversation_id}")
            
            # Get the central governance
            central_governance = await get_central_governance(user_id, conversation_id)
            
            # Create and initialize the integration
            integration = cls(user_id, conversation_id)
            await integration.initialize()
            
            # Register with governance
            await central_governance.register_agent(
                integration.agent_id,
                AgentType.CONFLICT_ANALYST,
                integration
            )
            
            logger.info(f"Successfully registered conflict system integration for user={user_id}, conversation={conversation_id}")
            
            return {"success": True, "integration": integration}
        except Exception as e:
            logger.error(f"Error registering conflict system integration: {e}")
            return {"success": False, "message": str(e)}

# Registration method remains the same
register_enhanced_integration = ConflictSystemIntegration.register_enhanced_integration
