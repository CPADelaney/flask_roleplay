# lore/lore_system.py

"""
Lore System - Main Entry Point (Refactored)

Now integrates all specialized managers (Education, Religion, LocalLore, etc.)
alongside existing data access and generator components, to serve as a true
'unified orchestrator' for all lore operations in the system.

IMPORTANT FIX: This module previously had a circular dependency with NyxUnifiedGovernor.
The circular dependency has been resolved by:
1. Adding a re-entry guard (_initializing flag) to prevent infinite recursion
2. Removing the get_central_governance() call from initialize()
3. Adding a set_governor() method for dependency injection
4. Having the governor set itself on the LoreSystem after creation

This ensures a clean, one-way initialization flow:
Governor → LoreSystem (not LoreSystem ↔ Governor)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# 1. Import specialized managers
from lore.managers.education import EducationalSystemManager
from lore.managers.religion import ReligionManager
from lore.managers.local_lore import LocalLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.systems.regional_culture import RegionalCultureSystem
from lore.managers.base_manager import BaseLoreManager
from lore.managers.politics import WorldPoliticsManager
from lore.systems.dynamics import LoreDynamicsSystem  # example if you want the 'dynamics system' too

# 2. Existing data-access and integration imports
from .data_access import NPCDataAccess, LocationDataAccess, FactionDataAccess, LoreKnowledgeAccess
from .integration import NPCLoreIntegration, ConflictIntegration, ContextEnhancer
from .lore_generator import DynamicLoreGenerator

# 3. Nyx governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance
from datetime import timedelta

logger = logging.getLogger(__name__)

# Cache for LoreSystem instances
LORE_SYSTEM_INSTANCES: Dict[str, "LoreSystem"] = {}

class LoreSystem:
    """
    Unified interface for *all* lore-related functionality, now referencing
    specialized managers for Education, Religion, LocalLore, Politics, etc.
    
    IMPORTANT: To avoid circular dependencies, this class no longer creates its own
    governor during initialization. Instead, the governor should be set externally
    via the set_governor() method after both objects are created.
    
    Typical usage:
        lore_system = LoreSystem.get_instance(user_id, conversation_id)
        governor = NyxUnifiedGovernor(user_id, conversation_id)
        lore_system.set_governor(governor)
        await lore_system.initialize()
    """

    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """Initialize the LoreSystem with optional user and conversation context."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.initialized = False
        self._initializing = False  # Guard against re-entry
        self.governor = None  # Will be injected, not created

        # 4. Built-in managers from your system
        self.npc_data = NPCDataAccess(user_id, conversation_id)
        self.location_data = LocationDataAccess(user_id, conversation_id)
        self.faction_data = FactionDataAccess(user_id, conversation_id)
        self.lore_knowledge = LoreKnowledgeAccess(user_id, conversation_id)

        self.npc_integration = NPCLoreIntegration(user_id, conversation_id)
        self.conflict_integration = ConflictIntegration(user_id, conversation_id)
        self.context_enhancer = ContextEnhancer(user_id, conversation_id)

        # 5. Additional specialized managers from your code base
        self.education_manager = EducationalSystemManager(user_id, conversation_id)
        self.religion_manager = ReligionManager(user_id, conversation_id)
        self.local_lore_manager = LocalLoreManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.regional_culture_system = RegionalCultureSystem(user_id, conversation_id)
        self.world_politics_manager = WorldPoliticsManager(user_id, conversation_id)

        # For the new "dynamics" system as well:
        self.lore_dynamics_system = LoreDynamicsSystem(user_id, conversation_id)

        # Store prohibited actions from directives, if any
        self.prohibited_actions = []
        # Store modifications from directives
        self.action_modifications = {}

    @classmethod
    def get_instance(cls, user_id: Optional[int] = None, conversation_id: Optional[int] = None) -> "LoreSystem":
        """Singleton instance retrieval for a given user/conversation context."""
        key = f"{user_id or 'global'}:{conversation_id or 'global'}"
        if key not in LORE_SYSTEM_INSTANCES:
            LORE_SYSTEM_INSTANCES[key] = cls(user_id, conversation_id)
        return LORE_SYSTEM_INSTANCES[key]

    def set_governor(self, governor: Any) -> None:
        """Set the governor instance. Should be called by the governor after creation."""
        self.governor = governor
        logger.info(f"[LoreSystem] Governor set for user {self.user_id}, conversation {self.conversation_id}")

    async def initialize(self, governor=None) -> bool:
        """Initialize the LoreSystem and all its components."""
        # Guard against re-entry
        if self.initialized or self._initializing:
            logger.debug("[LoreSystem] Already initialized or initializing, skipping")
            return True
    
        self._initializing = True
        try:
            # Use provided governor if available
            if governor:
                self.governor = governor
                logger.info("[LoreSystem] Using provided governor instance")
            
            logger.info("[LoreSystem] Starting initialization of data access components")
            
            # Check if governor is set
            if self.governor:
                logger.info("[LoreSystem] Governor already set, proceeding with initialization")
            else:
                logger.warning("[LoreSystem] No governor set yet, some features may be limited")
    
            # 6. Initialize the data access + integration components
            logger.info("[LoreSystem] Initializing: NPCDataAccess")
            await self.npc_data.initialize()
    
            logger.info("[LoreSystem] Initializing: LocationDataAccess")
            await self.location_data.initialize()
    
            logger.info("[LoreSystem] Initializing: FactionDataAccess")
            await self.faction_data.initialize()
    
            logger.info("[LoreSystem] Initializing: LoreKnowledgeAccess")
            await self.lore_knowledge.initialize()
    
            logger.info("[LoreSystem] Initializing: NPCLoreIntegration")
            # IMPORTANT: Set the governor on NPCLoreIntegration before initializing
            if hasattr(self.npc_integration, 'set_governor'):
                self.npc_integration.set_governor(self.governor)
            await self.npc_integration.initialize()
    
            logger.info("[LoreSystem] Initializing: ConflictIntegration")
            # Set governor if the component supports it
            if hasattr(self.conflict_integration, 'set_governor'):
                self.conflict_integration.set_governor(self.governor)
            await self.conflict_integration.initialize()

            logger.info("[LoreSystem] Initializing: DynamicLoreGenerator")
            # Get the instance with the governor to avoid circular dependency
            self.generator = DynamicLoreGenerator.get_instance(user_id, conversation_id, self.governor)
            await self.generator.initialize()
    
            logger.info("[LoreSystem] Initializing: ContextEnhancer")
            # Set governor if the component supports it
            if hasattr(self.context_enhancer, 'set_governor'):
                self.context_enhancer.set_governor(self.governor)
            await self.context_enhancer.initialize()
    
            logger.info("[LoreSystem] Initializing: DynamicLoreGenerator")
            await self.generator.initialize()
    
            # 7. Initialize additional integrated managers
            logger.info("[LoreSystem] Ensuring: EducationalSystemManager")
            # Pass governor if the manager supports it
            if hasattr(self.education_manager, 'set_governor'):
                self.education_manager.set_governor(self.governor)
            await self.education_manager.ensure_initialized()
    
            logger.info("[LoreSystem] Ensuring: ReligionManager")
            if hasattr(self.religion_manager, 'set_governor'):
                self.religion_manager.set_governor(self.governor)
            await self.religion_manager.ensure_initialized()
    
            logger.info("[LoreSystem] Ensuring: LocalLoreManager")
            if hasattr(self.local_lore_manager, 'set_governor'):
                self.local_lore_manager.set_governor(self.governor)
            await self.local_lore_manager.ensure_initialized()
    
            logger.info("[LoreSystem] Ensuring: GeopoliticalSystemManager")
            if hasattr(self.geopolitical_manager, 'set_governor'):
                self.geopolitical_manager.set_governor(self.governor)
            await self.geopolitical_manager.ensure_initialized()
    
            logger.info("[LoreSystem] Ensuring: RegionalCultureSystem")
            if hasattr(self.regional_culture_system, 'set_governor'):
                self.regional_culture_system.set_governor(self.governor)
            await self.regional_culture_system.ensure_initialized()
    
            logger.info("[LoreSystem] Ensuring: WorldPoliticsManager")
            if hasattr(self.world_politics_manager, 'set_governor'):
                self.world_politics_manager.set_governor(self.governor)
            await self.world_politics_manager.ensure_initialized()
    
            logger.info("[LoreSystem] Ensuring: LoreDynamicsSystem")
            if hasattr(self.lore_dynamics_system, 'set_governor'):
                self.lore_dynamics_system.set_governor(self.governor)
            await self.lore_dynamics_system.ensure_initialized()
    
            # Only register with governance if governor is set
            if self.governor:
                logger.info("[LoreSystem] Registering with Nyx governance")
                await self.register_with_governance()
            else:
                logger.warning("[LoreSystem] No governor set, skipping governance registration")
    
            self.initialized = True
            logger.info("[LoreSystem] Initialization successful.")
            return True
    
        except Exception as e:
            logger.exception(f"[LoreSystem] ERROR during initialization: {e}")
            self.initialized = False  # Ensure we're not marked as initialized on error
            return False
        finally:
            self._initializing = False
    
    # Also add this method to help with debugging:
    def is_initializing(self) -> bool:
        """Check if the LoreSystem is currently initializing."""
        return self._initializing
    
    def is_initialized(self) -> bool:
        """Check if the LoreSystem has been initialized."""
        return self.initialized

    async def register_with_governance(self):
        """Register the lore system with Nyx governance."""
        if not self.governor:
            logger.warning("[LoreSystem] Cannot register with governance - no governor set")
            return

        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_system",
            agent_instance=self
        )

        # Issue standard directives
        await self._issue_standard_directives()

    async def _issue_standard_directives(self):
        """Issue standard directives for the lore system."""
        if not self.governor:
            return

        # Example directive for lore generation
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Maintain world lore consistency and generate new lore as needed.",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24 * 60  # 24 hours
        )

        # Example directive for NPC lore integration
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="npc_lore_integration",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Ensure NPCs have appropriate lore knowledge based on their backgrounds.",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24 * 60
        )

    # ---------------------------------------------------------------------
    # NPC Lore Methods
    # ---------------------------------------------------------------------

    async def get_npc_lore_knowledge(self, npc_id: int) -> List[Dict[str, Any]]:
        """Get all lore known by an NPC."""
        return await self.lore_knowledge.get_entity_knowledge("npc", npc_id)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="initialize_npc_lore_knowledge",
        action_description="Initializing lore knowledge for NPC {npc_id}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def initialize_npc_lore_knowledge(self, ctx, npc_id: int,
                                            cultural_background: str,
                                            faction_affiliations: List[str]) -> Dict[str, Any]:
        """Initialize an NPC's lore knowledge based on background."""
        return await self.npc_integration.initialize_npc_lore_knowledge(
            ctx, npc_id, cultural_background, faction_affiliations
        )

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="process_npc_lore_interaction",
        action_description="Processing lore interaction for NPC {npc_id}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def process_npc_lore_interaction(self, ctx, npc_id: int, player_input: str) -> Dict[str, Any]:
        """Handle a lore interaction between the player and an NPC."""
        return await self.npc_integration.process_npc_lore_interaction(ctx, npc_id, player_input)

    # ---------------------------------------------------------------------
    # Location Lore Methods
    # ---------------------------------------------------------------------

    async def get_location_lore(self, location_id: int) -> Dict[str, Any]:
        """Get lore specific to a location."""
        return await self.location_data.get_location_with_lore(location_id)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_comprehensive_location_context",
        action_description="Getting comprehensive lore for location: {location_name}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def get_comprehensive_location_context(self, ctx, location_name: str) -> Dict[str, Any]:
        """Get a full lore context for a location, including culture/politics environment."""
        location = await self.location_data.get_location_by_name(location_name)
        if not location:
            return {}

        location_id = location.get("id")
        location_lore = await self.location_data.get_location_with_lore(location_id)
        cultural_info = await self.location_data.get_cultural_context_for_location(location_id)
        political_info = await self.location_data.get_political_context_for_location(location_id)
        environment_info = await self.location_data.get_environmental_conditions(location_id)

        return {
            "location": location,
            "lore": location_lore,
            "cultural_context": cultural_info,
            "political_context": political_info,
            "environment": environment_info
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_scene_description",
        action_description="Generating scene description for location: {location}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def generate_scene_description_with_lore(self, ctx, location: str) -> Dict[str, Any]:
        """Generate a scene description enhanced with relevant lore."""
        return await self.context_enhancer.generate_scene_description(location)

    # ---------------------------------------------------------------------
    # Education / Religion / Local Lore Accessors
    # ---------------------------------------------------------------------

    async def generate_educational_systems(self) -> List[Dict[str, Any]]:
        """
        High-level call to produce educational systems (or retrieve them)
        by delegating to the EducationalSystemManager.
        """
        # For example:
        return await self.education_manager.generate_educational_systems(None)

    async def generate_knowledge_traditions(self) -> List[Dict[str, Any]]:
        """
        Similarly, a high-level call to produce knowledge traditions
        from the EducationalSystemManager.
        """
        return await self.education_manager.generate_knowledge_traditions(None)

    async def generate_religion(self) -> Dict[str, Any]:
        """
        High-level call to produce a complete faith system
        from the ReligionManager.
        """
        return await self.religion_manager.generate_complete_faith_system(None)

    async def distribute_religions_across_nations(self) -> List[Dict[str, Any]]:
        """
        Use ReligionManager to distribute religions across world nations.
        """
        return await self.religion_manager.distribute_religions(None)

    async def generate_local_lore_for_location(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level call to produce local myths/histories/landmarks
        from the LocalLoreManager.
        """
        return await self.local_lore_manager.generate_location_lore(None, location_data)

    # ---------------------------------------------------------------------
    # Lore Generation & Evolution
    # ---------------------------------------------------------------------

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_lore",
        action_description="Generating complete lore for environment: {environment_desc}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def generate_complete_lore(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """Generate a complete set of lore for a game world via the generator."""
        return await self.generator.generate_complete_lore(environment_desc)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore based on event: {event_description}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def evolve_lore_with_event(self, ctx, event_description: str) -> Dict[str, Any]:
        """Update world lore based on a significant narrative event."""
        return await self.generator.evolve_lore_with_event(event_description)

    # Additionally, you could unify dynamic-lore updates with the LoreDynamicsSystem:
    async def mature_world_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        """Example: call the LoreDynamicsSystem to mature lore over time."""
        return await self.lore_dynamics_system.mature_lore_over_time(days_passed)
    
    # ---------------------------------------------------------------------
    # Canon Methods (NEW) - These are critical for the system to work
    # ---------------------------------------------------------------------
    
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
        This is the method that NyxUnifiedGovernor calls.

        Args:
            entity_type (str): The type of entity (e.g., 'Nations', 'Factions').
            entity_identifier (Dict): A dict to find the entity (e.g., {'id': 123}).
            updates (Dict): A dict of columns and their new values (e.g., {'leader_npc_id': 456}).
            reason (str): The narrative reason for the change.
        """
        # Import canon module
        from lore.core import canon
        from db.connection import get_db_connection_context
        
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
                        if hasattr(self, 'lore_dynamics_system'):
                            await self.lore_dynamics_system.evolve_lore_with_event(
                                f"A conflict arose: {reason}. Details: {', '.join(conflicts)}"
                            )
                        return {"status": "conflict_generated", "details": conflicts}

                    # Step 3b: No conflict, commit the change
                    set_clauses = [f"{key} = ${i+1}" for i, key in enumerate(updates.keys())]
                    set_sql = ", ".join(set_clauses)
                    # The entity identifier values come after the update values
                    update_values = list(updates.values()) + list(entity_identifier.values())
                    
                    update_query = f"UPDATE {entity_type} SET {set_sql} WHERE {where_sql}"
                    await conn.execute(update_query, *update_values)

                    # Step 4: Log the change as a canonical event in unified memory
                    event_text = f"The {entity_type} identified by {entity_identifier} was updated. Reason: {reason}. Changes: {updates}"
                    await canon.log_canonical_event(ctx, conn, event_text, tags=[entity_type.lower(), 'state_change'], significance=8)
            
            # Step 5: Propagate consequences to other systems (outside the transaction)
            if hasattr(self, 'lore_dynamics_system'):
                await self.lore_dynamics_system.evolve_lore_with_event(f"A world state change occurred: {reason}")
            
            return {"status": "committed", "entity_type": entity_type, "identifier": entity_identifier, "changes": updates}

        except Exception as e:
            logger.exception(f"Failed to enact change for {entity_type} ({entity_identifier}): {e}")
            return {"status": "error", "message": str(e)}
    
    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    async def cleanup(self):
        """Clean up resources used by the LoreSystem (and managers)."""
        try:
            # Collect all cleanup tasks in parallel
            cleanup_tasks = [
                self.npc_data.cleanup(),
                self.location_data.cleanup(),
                self.faction_data.cleanup(),
                self.lore_knowledge.cleanup(),
                self.npc_integration.cleanup(),
                self.conflict_integration.cleanup(),
                self.context_enhancer.cleanup(),
                self.generator.cleanup()
            ]
    
            # Example approach: if each manager implements a 'cleanup' method, call it
            for manager in [
                self.education_manager,
                self.religion_manager,
                self.local_lore_manager,
                self.geopolitical_manager,
                self.regional_culture_system,
                self.world_politics_manager,
                self.lore_dynamics_system
            ]:
                if hasattr(manager, "cleanup") and callable(manager.cleanup):
                    cleanup_tasks.append(manager.cleanup())
    
            # Execute all cleanup tasks concurrently
            import asyncio
            await asyncio.gather(*cleanup_tasks)
    
            # Remove this instance from the global cache
            key = f"{self.user_id or 'global'}:{self.conversation_id or 'global'}"
            if key in LORE_SYSTEM_INSTANCES:
                del LORE_SYSTEM_INSTANCES[key]
    
        except Exception as e:
            logger.error(f"Error during LoreSystem cleanup: {e}")
