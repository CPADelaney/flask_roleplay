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
from logic.fully_integrated_npc_system import IntegratedNPCSystem
from logic.conflict_system.conflict_agents import (
    triage_agent, conflict_generation_agent, stakeholder_agent,
    manipulation_agent, resolution_agent, ConflictContext, initialize_agents
)
from logic.conflict_system.conflict_tools import (
    get_active_conflicts, get_conflict_details, get_conflict_stakeholders,
    get_resolution_paths, get_player_involvement, get_internal_conflicts,
    update_conflict_progress, update_stakeholder_status, add_resolution_path,
    update_player_involvement, add_internal_conflict, resolve_internal_conflict,
    generate_conflict_consequences
)

logger = logging.getLogger(__name__)

class ConflictSystemIntegration:
    """
    Combined integration of conflict system and Nyx governance, including enhanced directive handling.
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

        # Directive handlers (can be overridden or used by enhanced handlers)
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
            self.agents = await initialize_agents()
            self.lore_system = await self.get_lore_system(self.user_id, self.conversation_id)
            self.npc_system = await self.get_npc_system(self.user_id, self.conversation_id)
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

    # -------------------------- GOVERNANCE/REGISTRATION -------------------------- #

    @staticmethod
    async def register_with_governance(user_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Register a (standard) ConflictSystemIntegration with Nyx governance.
        """
        try:
            governance = await get_central_governance(user_id, conversation_id)
            conflict_system = ConflictSystemIntegration(user_id, conversation_id)
            await conflict_system.initialize()
            await governance.register_agent(
                agent_type=AgentType.CONFLICT_ANALYST,
                agent_instance=conflict_system,
                agent_id="conflict_manager"
            )
            if hasattr(governance, "registered_agents"):
                governance.registered_agents[AgentType.CONFLICT_ANALYST] = conflict_system
            logger.info("Conflict System successfully registered with governance")
            return {"success": True, "message": "Registered conflict system."}
        except Exception as e:
            logger.error(f"Error registering conflict system: {e}", exc_info=True)
            return {"success": False, "message": str(e)}

    @staticmethod
    async def register_enhanced_integration(user_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Register an enhanced ConflictSystemIntegration (additional/overridden handlers).
        """
        try:
            governance = await get_central_governance(user_id, conversation_id)
            class EnhancedConflictSystem(ConflictSystemIntegration):
                pass  # customize if needed
            enhanced_system = EnhancedConflictSystem(user_id, conversation_id)
            await enhanced_system.initialize()
            await governance.register_agent(
                agent_type=AgentType.CONFLICT_ANALYST,
                agent_instance=enhanced_system,
                agent_id="conflict_analyst"
            )
            if hasattr(governance, "registered_agents"):
                governance.registered_agents[AgentType.CONFLICT_ANALYST] = enhanced_system
            logger.info("Enhanced Conflict System registered with Nyx governance")
            return {"success": True, "message": "Registered enhanced conflict system."}
        except Exception as e:
            logger.error(f"Error registering enhanced conflict system: {e}", exc_info=True)
            return {"success": False, "message": str(e)}

    @function_tool
    async def register_enhanced_conflict_system_tool(ctx: RunContextWrapper) -> Dict[str, Any]:
        """Agent tool: register the enhanced conflict system."""
        user_id = ctx.agent_context.get("user_id")
        conversation_id = ctx.agent_context.get("conversation_id")
        if not user_id or not conversation_id:
            return {"success": False, "message": "Missing user_id or conversation_id in context"}
        return await ConflictSystemIntegration.register_enhanced_integration(user_id, conversation_id)

    # -------------------------- STORY CONTEXT / DATA -------------------------- #

    async def _get_story_context(self) -> Dict[str, Any]:
        """
        Gather everything needed to resume or display the current story:
          - active NPCs
          - lore elements for the current narrative
          - story progression details
          - the narrative ID itself
        """
        from npcs.npc_agent_system import get_active_npcs
    
        # Prepare a default response
        default = {
            "active_npcs": [],
            "lore_context": [],
            "story_progression": {},
            "current_narrative_id": None,
        }
    
        try:
            active_npcs = await get_active_npcs()
    
            narrative_id = await self._get_current_narrative_id()
            lore_context = (
                await self.lore_system.get_narrative_elements(narrative_id)
                if narrative_id
                else []
            )
    
            story_progression = await self._get_story_progression()
    
            return {
                "active_npcs": active_npcs,
                "lore_context": lore_context,
                "story_progression": story_progression,
                "current_narrative_id": narrative_id,
            }
    
        except Exception as e:
            logger.error(f"Error getting story context: {e}")
            return default

    async def _get_current_narrative_id(self) -> Optional[int]:
        try:
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
        try:
            async with get_db_connection_context() as conn:
                day_row = await conn.fetchrow(
                    "SELECT value FROM CurrentRoleplay WHERE key = 'CurrentDay' AND user_id = $1 AND conversation_id = $2",
                    self.user_id, self.conversation_id
                )
                current_day = int(day_row['value']) if day_row and day_row['value'] else 1
                progression_row = await conn.fetchrow(
                    "SELECT progress, phase, key_decisions FROM StoryProgression WHERE user_id = $1 AND conversation_id = $2",
                    self.user_id, self.conversation_id
                )
                if progression_row:
                    return {
                        'current_day': current_day,
                        'progress': progression_row['progress'],
                        'phase': progression_row['phase'],
                        'key_decisions': progression_row['key_decisions']
                    }
                return {'current_day': current_day}
        except Exception as e:
            logger.error(f"Error getting story progression: {e}")
            return {'current_day': 1}

    # ------------------ DIRECTIVE HANDLING (including ENHANCED) ------------------ #

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
            # Enhanced/extended handling example below; customize if you want "enhanced" behaviors as in v2.
            return {"success": False, "message": f"Unhandled directive type: {directive_type}", "directive_type": directive_type}

    async def _handle_action_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handles action requests from governance."""
        action_type = directive_data.get("action_type")
        action_params = directive_data.get("parameters", {})
        # Augmented example: optionally augment params for enhanced context/features.
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

        # Example enhancement: auto background conflict for certain scenes
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
        return {"success": True, "message": "Conflict system is aware of scene change", "scene_type": scene_type, "location": location}

    async def _handle_prohibition_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        prohibited_action = directive_data.get("prohibited_action")
        reason = directive_data.get("reason")
        duration = directive_data.get("duration_minutes", 60)
        logger.info(f"Registering prohibition: {prohibited_action} for reason: {reason}, duration: {duration}")
        # Optionally, record tension or adjust available actions in the world
        return {"success": True, "message": f"Prohibition registered: {prohibited_action}", "duration_minutes": duration}

    async def _handle_information_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        info_type = directive_data.get("info_type")
        info_content = directive_data.get("content")
        logger.info(f"Processing information: type={info_type}")
        # Enhanced info: auto escalate, update, or notify as needed
        return {"success": True, "message": f"Information processed: {info_type}", "info_type": info_type}

    # ------------------- PERMISSION, REPORTING, REQUESTS ------------------- #

    async def check_permission(self, action_type: str, action_details: Dict[str, Any]) -> Dict[str, Any]:
        governance = await get_central_governance(self.user_id, self.conversation_id)
        permission = await governance.check_action_permission(
            agent_type=AgentType.CONFLICT_ANALYST, agent_id=self.agent_id,
            action_type=action_type, action_details=action_details
        )
        return permission

    async def report_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        governance = await get_central_governance(self.user_id, self.conversation_id)
        return await governance.process_agent_action_report(
            agent_type=AgentType.CONFLICT_ANALYST, agent_id=self.agent_id,
            action=action, result=result
        )

    async def process_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        await self.initialize()
        if request_type == "generate_conflict":
            return await self.generate_conflict(request_data)
        elif request_type == "resolve_conflict":
            return await self.resolve_conflict(request_data)
        elif request_type == "update_stakeholders":
            return await self.update_stakeholders(request_data)
        elif request_type == "manage_manipulation":
            return await self.manage_manipulation(request_data)
        else:
            return {"success": False, "message": f"Unknown request type: {request_type}"}

    # ------------------- CONFLICT MANAGEMENT AND RESOLUTION ------------------- #

    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="generate_conflict",
        action_description="Generate a new conflict with stakeholders"
    )
    async def generate_conflict(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        await self.initialize()
        try:
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

            # Consequence handling is as in your v1, call self._apply_conflict_rewards() etc.
            # Adapt if you want to deeper process or summarize
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
            return {
                "success": True,
                "manipulation_result": result
            }
        except Exception as e:
            logger.error(f"Error managing manipulation: {str(e)}", exc_info=True)
            return {"success": False, "message": str(e)}

    async def _apply_conflict_rewards(self, consequences: List[Dict[str, Any]]) -> None:
        """
        Apply rewards/consequences from conflict resolution: stats, inventory, perks, relationships, etc.
        """
        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    for consequence in consequences:
                        ctype = consequence.get("type")
                        # Player stat changes
                        if ctype == "player_stat" and "stat_changes" in consequence:
                            for stat, change in consequence["stat_changes"].items():
                                query = f"""
                                    UPDATE PlayerStats
                                    SET {stat} = {stat} + $1
                                    WHERE user_id = $2 AND conversation_id = $3
                                """
                                await conn.execute(query, change, self.user_id, self.conversation_id)
                                logger.info(f"Updated player stat {stat} by {change} for user {self.user_id}")
                        # Item rewards
                        elif ctype == "item_reward" and "item" in consequence:
                            item = consequence["item"]
                            query = """
                                INSERT INTO PlayerInventory
                                (user_id, conversation_id, item_name, item_description,
                                 item_category, item_properties, quantity, equipped)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """
                            await conn.execute(
                                query,
                                self.user_id,
                                self.conversation_id,
                                item["name"], item["description"], item.get("category", "conflict_reward"),
                                json.dumps({
                                    "rarity": item.get("rarity", "common"),
                                    "resolution_style": item.get("resolution_style", "neutral"),
                                    "source": "conflict_resolution"
                                }),
                                1, False
                            )
                            logger.info(f"Added item {item['name']} to inventory for user {self.user_id}")
                        # Perk rewards
                        elif ctype == "perk_reward" and "perk" in consequence:
                            perk = consequence["perk"]
                            query = """
                                SELECT perk_id FROM PlayerPerks
                                WHERE user_id = $1 AND conversation_id = $2 AND perk_name = $3
                            """
                            existing = await conn.fetchrow(query, self.user_id, self.conversation_id, perk["name"])
                            if existing:
                                update_q = """
                                    UPDATE PlayerPerks
                                    SET perk_tier = GREATEST(perk_tier, $1), perk_description = $2
                                    WHERE user_id = $3 AND conversation_id = $4 AND perk_name = $5
                                """
                                await conn.execute(update_q,
                                    perk.get("tier", 1), perk["description"],
                                    self.user_id, self.conversation_id, perk["name"]
                                )
                            else:
                                insert_q = """
                                    INSERT INTO PlayerPerks
                                    (user_id, conversation_id, perk_name, perk_description,
                                     perk_category, perk_tier, perk_properties)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                                """
                                await conn.execute(insert_q,
                                    self.user_id, self.conversation_id,
                                    perk["name"], perk["description"],
                                    perk.get("category", "conflict_resolution"), perk.get("tier", 1),
                                    json.dumps({
                                        "resolution_style": perk.get("resolution_style", "neutral"),
                                        "source": "conflict_resolution"
                                    })
                                )
                            logger.info(f"Added/updated perk {perk['name']} for user {self.user_id}")
                        # Special rewards
                        elif ctype == "special_reward" and "special_reward" in consequence:
                            special = consequence["special_reward"]
                            query = """
                                INSERT INTO PlayerSpecialRewards
                                (user_id, conversation_id, reward_name, reward_description, 
                                 reward_effect, reward_category, reward_properties, used)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """
                            await conn.execute(
                                query, self.user_id, self.conversation_id, special["name"],
                                special["description"], special.get("effect", ""),
                                special.get("category", "unique_conflict_reward"),
                                json.dumps({
                                    "resolution_style": special.get("resolution_style", "neutral"),
                                    "source": "major_conflict_resolution"
                                }),
                                False
                            )
                            logger.info(f"Added special reward {special['name']} for user {self.user_id}")
                        # NPC relationship changes
                        elif ctype == "npc_relationship" and consequence.get("npc_id"):
                            npc_id = consequence.get("npc_id")
                            rel_changes = consequence.get("relationship_change", {})
                            for rel_type, amount in rel_changes.items():
                                if amount == 0: continue
                                query = f"""
                                    UPDATE NPCStats
                                    SET {rel_type} = {rel_type} + $1
                                    WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
                                """
                                await conn.execute(query, amount, npc_id, self.user_id, self.conversation_id)
                            logger.info(f"Updated relationship with NPC {npc_id} for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error applying conflict rewards: {str(e)}", exc_info=True)
            raise

    async def _extract_conflict_details(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        # Basic details
        details = {
            "id": conflict_data.get("id"),
            "name": conflict_data.get("name", "Unnamed Conflict"),
            "type": conflict_data.get("conflict_type", "standard"),
            "status": conflict_data.get("status", "inactive"),
            "location": conflict_data.get("location", "Unknown"),
            "description": conflict_data.get("description", ""),
            "phase": conflict_data.get("phase", "brewing"),
            "faction_a": conflict_data.get("faction_a_name", ""),
            "faction_b": conflict_data.get("faction_b_name", ""),
            "resolution_path": conflict_data.get("resolution_path", ""),
            "completion_status": conflict_data.get("completion_status", "incomplete")
        }
        details["is_resolved"] = details["status"] == "resolved"
        phases = ["brewing", "active", "climax", "resolved"]
        phase_idx = phases.index(details["phase"]) if details["phase"] in phases else 0
        details["progress"] = (phase_idx / (len(phases) - 1)) * 100  # As percentage
        return details

    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="get_conflicts_by_location",
        action_description="Getting conflicts for location: {location}"
    )
    async def get_conflicts_by_location(self, location: str) -> List[Dict[str, Any]]:
        """
        Get all conflicts associated with a specific location.
        """
        try:
            await self.initialize()
            async with get_db_connection_context() as conn:
                location_id = await conn.fetchval("""
                    SELECT id FROM Locations 
                    WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                """, location, self.user_id, self.conversation_id)
                rows = await conn.fetch("""
                    SELECT c.*, cp.resolution_path, cp.completion_status
                    FROM Conflicts c
                    LEFT JOIN ConflictPaths cp ON c.id = cp.conflict_id 
                        AND cp.user_id = c.user_id 
                        AND cp.conversation_id = c.conversation_id
                    WHERE c.user_id = $1 AND c.conversation_id = $2
                    AND (c.location = $3 OR c.location_id = $4)
                    ORDER BY c.created_at DESC
                """, self.user_id, self.conversation_id, location, location_id)
                conflicts = []
                for row in rows:
                    conflict_data = dict(row)
                    details = await self._extract_conflict_details(conflict_data)
                    conflicts.append(details)
                return conflicts
        except Exception as e:
            logger.error(f"Error getting conflicts by location: {e}", exc_info=True)
            return []

    async def resolve_conflict_with_lore(self, conflict_id: int) -> Dict[str, Any]:
        """
        Resolve conflict with lore system integration.
        """
        try:
            run_ctx = RunContextWrapper({
                "user_id": self.user_id, "conversation_id": self.conversation_id
            })
            conflict_data = await get_conflict_details(run_ctx, conflict_id)
            stakeholders = await get_conflict_stakeholders(run_ctx, conflict_id)
            resolution = await self.generate_conflict_resolution(conflict_id, stakeholders)
            narrative_result = await self.lore_system.handle_narrative_event(
                run_ctx,
                f"Conflict resolution: {conflict_data.get('description', '')}",
                affected_lore_ids=conflict_data.get('affected_lore_ids', []),
                resolution_type=resolution['type'],
                impact_level=resolution['impact_level']
            )
            await self.update_relationships_from_resolution(conflict_id, resolution, stakeholders)
            memory_system = await self._get_memory_system()
            result = {
                'conflict_id': conflict_id,
                'resolution': resolution,
                'narrative_impact': narrative_result,
                'affected_stakeholders': stakeholders,
                'timestamp': datetime.utcnow().isoformat()
            }
            await memory_system.add_memory(
                run_ctx,
                f"Conflict resolution for conflict {conflict_id}",
                result,
                memory_type="conflict_resolution"
            )
            return result
        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_id}: {e}", exc_info=True)
            return {'error': str(e), 'conflict_id': conflict_id, 'status': 'failed'}

    async def _get_memory_system(self):
        from memory.wrapper import MemorySystem
        return await MemorySystem.get_instance(self.user_id, self.conversation_id)

    async def generate_conflict_resolution(self, conflict_id: int, stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a conflict resolution proposal using agent + context.
        """
        try:
            conflict_context = ConflictContext(self.user_id, self.conversation_id)
            conflict = await get_conflict_details(RunContextWrapper({'user_id': self.user_id, 'conversation_id': self.conversation_id}), conflict_id)
            resolution_data = {
                "conflict_id": conflict_id,
                "conflict_details": conflict,
                "stakeholders": stakeholders,
                "resolution_type": "negotiated",
                "player_involvement": conflict.get("player_involvement", {})
            }
            resolution_result = await Runner.run(
                self.agents["resolution_agent"],
                json.dumps(resolution_data),
                context=conflict_context,
            )
            result = getattr(resolution_result, 'final_output', resolution_result)
            return {
                "type": "negotiated",
                "description": result.get("description", ""),
                "impact_level": "medium",
                "progress_value": result.get("progress_value"),
                "new_progress": result.get("new_progress"),
                "is_completed": result.get("is_completed")
            }
        except Exception as e:
            logger.error(f"Error generating conflict resolution: {e}")
            return {
                "type": "failed",
                "description": f"Failed to generate resolution: {str(e)}",
                "impact_level": "low"
            }

    async def update_relationships_from_resolution(self, conflict_id: int, resolution: Dict[str, Any], stakeholders: List[Dict[str, Any]]) -> None:
        """
        Update NPC relationships based on conflict resolution.
        """
        try:
            for stakeholder in stakeholders:
                npc_id = stakeholder.get("npc_id")
                if not npc_id:
                    continue
                # Calculate changes
                changes = {"trust": 0, "respect": 0, "closeness": 0, "tension": 0}
                rtype = resolution.get("type")
                if rtype == "negotiated":
                    changes["trust"] += 10
                    changes["respect"] += 5
                    changes["tension"] -= 20
                elif rtype == "forced":
                    changes["respect"] += 5
                    changes["tension"] -= 10
                elif rtype == "failed":
                    changes["tension"] += 10
                await self._apply_npc_relationship_changes(npc_id, changes)
        except Exception as e:
            logger.error(f"Error updating relationships from resolution: {e}")

    async def _apply_npc_relationship_changes(self, npc_id: int, changes: Dict[str, int]) -> None:
        try:
            async with get_db_connection_context() as conn:
                for stat, change in changes.items():
                    if change == 0:
                        continue
                    await conn.execute(f"""
                        UPDATE NPCStats
                        SET {stat} = GREATEST(0, LEAST(100, {stat} + $1))
                        WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
                    """, change, npc_id, self.user_id, self.conversation_id)
        except Exception as e:
            logger.error(f"Error applying NPC relationship changes: {e}")
            
register_enhanced_integration = ConflictSystemIntegration.register_enhanced_integration
