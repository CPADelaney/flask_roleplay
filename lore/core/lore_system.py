# lore/core/lore_system.py
"""
The unified LoreSystem - Primary implementation.
This is the core module that contains all lore-related logic and operations.
"""
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

from agents import RunContextWrapper
from lore.core.registry import ManagerRegistry
from lore.core import canon
from db.connection import get_db_connection_context
from nyx.nyx_governance import AgentType
from nyx.governance_helpers import with_governance

# Import all managers
from lore.managers.education import EducationalSystemManager
from lore.managers.religion import ReligionManager
from lore.managers.local_lore import LocalLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.systems.regional_culture import RegionalCultureSystem
from lore.managers.base_manager import BaseLoreManager
from lore.managers.politics import WorldPoliticsManager
from lore.systems.dynamics import LoreDynamicsSystem

# Import data access and integration components
from lore.data_access import NPCDataAccess, LocationDataAccess, FactionDataAccess, LoreKnowledgeAccess
from lore.integration import NPCLoreIntegration, ConflictIntegration, ContextEnhancer
from lore.lore_generator import DynamicLoreGenerator

logger = logging.getLogger(__name__)

class LoreSystem:
    """
    The core unified LoreSystem that handles all lore-related functionality.
    This is the primary implementation containing all logic.
    """
    _instances: Dict[str, "LoreSystem"] = {}

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.registry = ManagerRegistry(user_id, conversation_id)
        self.initialized = False
        self._initializing = False
        self.governor = None
        
        # Initialize all data access components
        self.npc_data = NPCDataAccess(user_id, conversation_id)
        self.location_data = LocationDataAccess(user_id, conversation_id)
        self.faction_data = FactionDataAccess(user_id, conversation_id)
        self.lore_knowledge = LoreKnowledgeAccess(user_id, conversation_id)
        
        # Initialize integration components
        self.npc_integration = NPCLoreIntegration(user_id, conversation_id)
        self.conflict_integration = ConflictIntegration(user_id, conversation_id)
        self.context_enhancer = ContextEnhancer(user_id, conversation_id)
        
        # Initialize specialized managers
        self.education_manager = EducationalSystemManager(user_id, conversation_id)
        self.religion_manager = ReligionManager(user_id, conversation_id)
        self.local_lore_manager = LocalLoreManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.regional_culture_system = RegionalCultureSystem(user_id, conversation_id)
        self.world_politics_manager = WorldPoliticsManager(user_id, conversation_id)
        self.lore_dynamics_system = LoreDynamicsSystem(user_id, conversation_id)
        
        # Track initialization state
        self.prohibited_actions = []
        self.action_modifications = {}
        self._managers_needing_registration = []

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

    def set_governor(self, governor: Any) -> None:
        """Set the governor instance. Should be called by the governor after creation."""
        self.governor = governor
        logger.info(f"[LoreSystem] Governor set for user {self.user_id}, conversation {self.conversation_id}")

    async def initialize(self, governor=None) -> bool:
        """Initialize the LoreSystem and all its components."""
        if self.initialized or self._initializing:
            logger.debug("[LoreSystem] Already initialized or initializing, skipping")
            return True
    
        self._initializing = True
        try:
            if governor:
                self.governor = governor
                logger.info("[LoreSystem] Using provided governor instance")
            
            logger.info("[LoreSystem] Starting initialization of data access components")
            
            # Initialize all components without governance registration
            await self._initialize_components()
            
            # Register with governance if available
            if self.governor:
                await self._register_all_with_governance()
            
            self.initialized = True
            logger.info("[LoreSystem] Initialization successful.")
            return True
    
        except Exception as e:
            logger.exception(f"[LoreSystem] ERROR during initialization: {e}")
            self.initialized = False
            return False
        finally:
            self._initializing = False

    async def _initialize_components(self):
        """Initialize all components WITHOUT governance registration."""
        # Initialize data access components
        logger.info("[LoreSystem] Initializing: NPCDataAccess")
        await self.npc_data.initialize()

        logger.info("[LoreSystem] Initializing: LocationDataAccess")
        await self.location_data.initialize()

        logger.info("[LoreSystem] Initializing: FactionDataAccess")
        await self.faction_data.initialize()

        logger.info("[LoreSystem] Initializing: LoreKnowledgeAccess")
        await self.lore_knowledge.initialize()

        # Initialize integration components with governor
        logger.info("[LoreSystem] Initializing: NPCLoreIntegration")
        if hasattr(self.npc_integration, 'set_governor'):
            self.npc_integration.set_governor(self.governor)
        await self.npc_integration.initialize()
        self._managers_needing_registration.append(('npc_integration', self.npc_integration))

        logger.info("[LoreSystem] Initializing: ConflictIntegration")
        if hasattr(self.conflict_integration, 'set_governor'):
            self.conflict_integration.set_governor(self.governor)
        await self.conflict_integration.initialize()
        self._managers_needing_registration.append(('conflict_integration', self.conflict_integration))

        logger.info("[LoreSystem] Initializing: ContextEnhancer")
        if hasattr(self.context_enhancer, 'set_governor'):
            self.context_enhancer.set_governor(self.governor)
        await self.context_enhancer.initialize()
        self._managers_needing_registration.append(('context_enhancer', self.context_enhancer))

        # Initialize generator
        logger.info("[LoreSystem] Initializing: DynamicLoreGenerator")
        self.generator = DynamicLoreGenerator.get_instance(self.user_id, self.conversation_id, self.governor)
        await self.generator.initialize()
        self._managers_needing_registration.append(('generator', self.generator))

        # Initialize specialized managers
        await self._initialize_specialized_managers()

    async def _initialize_specialized_managers(self):
        """Initialize specialized managers with timeout protection."""
        managers = [
            ("EducationalSystemManager", self.education_manager),
            ("ReligionManager", self.religion_manager),
            ("LocalLoreManager", self.local_lore_manager),
            ("GeopoliticalSystemManager", self.geopolitical_manager),
            ("RegionalCultureSystem", self.regional_culture_system),
            ("WorldPoliticsManager", self.world_politics_manager),
            ("LoreDynamicsSystem", self.lore_dynamics_system)
        ]
        
        for manager_name, manager in managers:
            logger.info(f"[LoreSystem] Ensuring: {manager_name}")
            
            if hasattr(manager, 'set_governor'):
                manager.set_governor(self.governor)
            
            try:
                await asyncio.wait_for(
                    manager.ensure_initialized(),
                    timeout=15.0
                )
                self._managers_needing_registration.append((manager_name.lower(), manager))
                logger.info(f"[LoreSystem] {manager_name} initialized successfully")
                
            except asyncio.TimeoutError:
                logger.error(f"[LoreSystem] {manager_name} init timed-out")
                raise RuntimeError(f"{manager_name} initialization timed out")
            except Exception as e:
                logger.error(f"[LoreSystem] Error initializing {manager_name}: {e}")
                raise

    async def _register_all_with_governance(self):
        """Register all components with governance after they're initialized."""
        logger.info("[LoreSystem] Registering all components with Nyx governance")
        
        # Register the LoreSystem itself
        await self.register_with_governance()
        
        # Register all managers
        for manager_id, manager in self._managers_needing_registration:
            try:
                if hasattr(manager, 'register_with_governance_deferred'):
                    await manager.register_with_governance_deferred()
                elif hasattr(manager, '_get_agent_type') and hasattr(manager, '_get_agent_id'):
                    await self.governor.register_agent(
                        agent_type=manager._get_agent_type(),
                        agent_id=manager._get_agent_id(),
                        agent_instance=manager
                    )
                    logger.info(f"[LoreSystem] Registered {manager_id} with governance")
            except Exception as e:
                logger.warning(f"[LoreSystem] Failed to register {manager_id} with governance: {e}")

    async def register_with_governance(self):
        """Register the lore system with Nyx governance."""
        if not self.governor:
            logger.warning("[LoreSystem] Cannot register with governance - no governor set")
            return

        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_system",
            agent_instance=self
        )

        await self._issue_standard_directives()

    async def _issue_standard_directives(self):
        """Issue standard directives for the lore system."""
        if not self.governor:
            return

        from nyx.nyx_governance import DirectiveType, DirectivePriority

        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Maintain world lore consistency and generate new lore as needed.",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24 * 60
        )

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

    def is_initializing(self) -> bool:
        """Check if the LoreSystem is currently initializing."""
        return self._initializing
    
    def is_initialized(self) -> bool:
        """Check if the LoreSystem has been initialized."""
        return self.initialized

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
    
                    # Step 2: Validate for conflicts (with improved JSON handling)
                    conflicts = []
                    for field, new_value in updates.items():
                        if field in existing_entity:
                            current_value = existing_entity[field]
                            
                            # Identify JSON fields based on common patterns
                            json_fields = [
                                'relationships', 'personality_traits', 'hobbies', 'likes', 
                                'dislikes', 'affiliations', 'schedule', 'archetypes', 
                                'memory', 'personality_patterns', 'trauma_triggers', 
                                'flashback_triggers', 'revelation_plan', 'group_data',
                                'mask_slippage_events', 'evolution_events', 'current_state',
                                'last_decision', 'item_properties', 'perk_properties',
                                'reward_properties', 'entry_metadata', 'tags', 'goal',
                                'leverage_used', 'metadata', 'network_data', 'timeline',
                                'primary_actors', 'diplomatic_events', 'military_events',
                                'civilian_impact', 'resolution_scenarios', 'most_likely_outcome',
                                'public_opinion', 'relations', 'power_centers', 'resolution_attempts',
                                'link_history', 'dynamics', 'experienced_crossroads',
                                'experienced_rituals', 'evolution_history', 'alliances',
                                'rivalries', 'requirements', 'stakeholders_involved',
                                'key_challenges', 'involved_npcs', 'serialized_state',
                                'event_payload', 'current_goals', 'predicted_futures',
                                'message', 'directive', 'message_content', 'response_content',
                                'governance_policy', 'theme_directives', 'pacing_directives',
                                'character_directives', 'payload', 'kink_profile',
                                'decision_meta', 'capabilities', 'state_data', 'metrics',
                                'error_log', 'learned_patterns', 'npc_names', 'trigger_context',
                                'context_data', 'changes'
                            ]
                            
                            if field in json_fields:
                                try:
                                    # Parse current value
                                    if isinstance(current_value, str):
                                        current_obj = json.loads(current_value) if current_value else {}
                                    else:
                                        current_obj = current_value if current_value is not None else {}
                                    
                                    # Parse new value
                                    if isinstance(new_value, str):
                                        new_obj = json.loads(new_value) if new_value else {}
                                    else:
                                        new_obj = new_value if new_value is not None else {}
                                    
                                    # Special handling for arrays
                                    if isinstance(current_obj, list) and isinstance(new_obj, list):
                                        # Don't conflict when adding to empty array
                                        if len(current_obj) == 0 and len(new_obj) > 0:
                                            continue
                                        # Don't conflict if new array contains all items from current
                                        if all(item in new_obj for item in current_obj):
                                            continue
                                    
                                    # Special handling for dicts
                                    if isinstance(current_obj, dict) and isinstance(new_obj, dict):
                                        # Don't conflict when adding to empty dict
                                        if len(current_obj) == 0 and len(new_obj) > 0:
                                            continue
                                        # Don't conflict if new dict is a superset
                                        if all(k in new_obj and new_obj[k] == v for k, v in current_obj.items()):
                                            continue
                                    
                                    # Compare as objects
                                    if current_obj != new_obj:
                                        conflict_detail = (
                                            f"Conflict on field '{field}'. "
                                            f"Current value: '{json.dumps(current_obj)}'. "
                                            f"Proposed value: '{json.dumps(new_obj)}'."
                                        )
                                        conflicts.append(conflict_detail)
                                        logger.warning(f"Conflict detected for {entity_type} ({entity_identifier}): {conflict_detail}")
                                except Exception as e:
                                    logger.debug(f"Error comparing JSON field {field}: {e}")
                                    if current_value != new_value:
                                        conflict_detail = (
                                            f"Conflict on field '{field}'. "
                                            f"Current value: '{current_value}'. "
                                            f"Proposed value: '{new_value}'."
                                        )
                                        conflicts.append(conflict_detail)
                            else:
                                # Regular field comparison
                                if current_value is not None and current_value != new_value:
                                    conflict_detail = (
                                        f"Conflict on field '{field}'. "
                                        f"Current value: '{existing_entity[field]}'. "
                                        f"Proposed value: '{new_value}'."
                                    )
                                    conflicts.append(conflict_detail)
                                    logger.warning(f"Conflict detected for {entity_type} ({entity_identifier}): {conflict_detail}")
    
                    if conflicts:
                        # Handle conflict by generating a new narrative event
                        dynamics = await self.registry.get_lore_dynamics()
                        conflict_description = f"A conflict arose: {reason}. Details: {', '.join(conflicts)}"
                        
                        if hasattr(dynamics, 'evolve_lore_with_event'):
                            await dynamics.evolve_lore_with_event(ctx, conflict_description)
                        else:
                            await dynamics(ctx, conflict_description)
                            
                        return {"status": "conflict_generated", "details": conflicts}
    
                    # Step 3: No conflict, commit the change
                    # Add type conversion for known boolean columns
                    boolean_columns = ['introduced', 'is_active', 'is_consolidated', 'is_archived']
                    
                    # Build SET clauses with proper casting
                    set_clauses = []
                    update_values = []
                    
                    for i, (key, value) in enumerate(updates.items()):
                        if key in boolean_columns:
                            bool_value = bool(value) if not isinstance(value, bool) else value
                            set_clauses.append(f"{key} = ${i+1}::boolean")
                            update_values.append(bool_value)
                        else:
                            set_clauses.append(f"{key} = ${i+1}")
                            update_values.append(value)
                    
                    set_sql = ", ".join(set_clauses)
                    
                    # Build WHERE clause
                    where_clauses_update = []
                    num_updates = len(updates)
                    for i, key in enumerate(entity_identifier.keys()):
                        placeholder_num = num_updates + i + 1
                        where_clauses_update.append(f"{key} = ${placeholder_num}")
                    where_sql_update = " AND ".join(where_clauses_update)
                    
                    update_values.extend(list(entity_identifier.values()))
                    
                    update_query = f"UPDATE {entity_type} SET {set_sql} WHERE {where_sql_update}"
                    await conn.execute(update_query, *update_values)
    
                    # Step 4: Log the change
                    event_text = f"The {entity_type} identified by {entity_identifier} was updated. Reason: {reason}. Changes: {updates}"
                    await canon.log_canonical_event(ctx, conn, event_text, tags=[entity_type.lower(), 'state_change'], significance=8)
            
            # Step 5: Propagate consequences
            event_description = self._create_detailed_event_description(
                entity_type, entity_identifier, updates, reason
            )
            
            if event_description:
                dynamics = await self.registry.get_lore_dynamics()
                
                if hasattr(dynamics, 'evolve_lore_with_event'):
                    await dynamics.evolve_lore_with_event(ctx, event_description)
                else:
                    await dynamics(ctx, event_description)
            
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
        """
        significant_fields = {
            'NPCStats': ['power_level', 'status', 'loyalty'],
            'Factions': ['leader_npc_id', 'power_level', 'territory', 'allies', 'rivals'],
            'Nations': ['leader_npc_id', 'government_type', 'stability'],
            'Locations': ['controlling_faction', 'strategic_value'],
            'NPCs': ['status', 'faction_affiliation', 'current_location']
        }
        
        if entity_type in significant_fields:
            updated_significant_fields = [
                field for field in significant_fields[entity_type] 
                if field in updates
            ]
            if not updated_significant_fields:
                return None
        
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
            event_parts.append(f"The {entity_type.lower()} structure has undergone important changes")
        
        event_parts.append(f"This occurred because: {reason}")
        
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
        
        if entity_type in ["Nations", "Factions"]:
            event_parts.append("This shift in power will ripple through the political landscape")
        elif entity_type == "Locations":
            event_parts.append("This territorial change may spark new conflicts or opportunities")
        elif entity_type == "NPCStats":
            event_parts.append("This personal transformation will influence their relationships")
        
        return ". ".join(event_parts)

    # --- Convenience Wrappers ---
    
    async def execute_coup(self, ctx, nation_id: int, new_leader_id: int, reason: str):
        """A high-level wrapper for a coup event."""
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
        if isinstance(ctx, str) or not hasattr(ctx, 'context'):
            ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
        
        location_id = None
        async with get_db_connection_context() as conn:
            location_id = await canon.find_or_create_location(ctx, conn, location_name)

        return await self.propose_and_enact_change(
            ctx=ctx,
            entity_type="Factions",
            entity_identifier={"id": faction_id},
            updates={"territory": location_name},
            reason=reason
        )

    # --- NPC Lore Methods ---
    
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

    # --- Location Lore Methods ---
    
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
        """Get a full lore context for a location."""
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

    # --- Education / Religion / Local Lore Accessors ---
    
    async def generate_educational_systems(self) -> List[Dict[str, Any]]:
        """Generate educational systems."""
        return await self.education_manager.generate_educational_systems(None)

    async def generate_knowledge_traditions(self) -> List[Dict[str, Any]]:
        """Generate knowledge traditions."""
        return await self.education_manager.generate_knowledge_traditions(None)

    async def generate_religion(self) -> Dict[str, Any]:
        """Generate a complete faith system."""
        return await self.religion_manager.generate_complete_faith_system(None)

    async def distribute_religions_across_nations(self) -> List[Dict[str, Any]]:
        """Distribute religions across world nations."""
        return await self.religion_manager.distribute_religions(None)

    async def generate_local_lore_for_location(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate local myths/histories/landmarks."""
        ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        return await self.local_lore_manager._generate_location_lore_impl(ctx, location_data)

    # --- Lore Generation & Evolution ---
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_lore",
        action_description="Generating complete lore for environment: {environment_desc}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def generate_complete_lore(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """Generate a complete set of lore for a game world."""
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

    async def mature_world_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        """Call the LoreDynamicsSystem to mature lore over time."""
        ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        return await self.lore_dynamics_system.mature_lore_over_time.fn(ctx, days_passed)

    # --- Cleanup ---
    
    async def cleanup(self):
        """Clean up resources used by the LoreSystem."""
        try:
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
    
            await asyncio.gather(*cleanup_tasks)
    
            key = f"{self.user_id}:{self.conversation_id}"
            if key in self._instances:
                del self._instances[key]
    
        except Exception as e:
            logger.error(f"Error during LoreSystem cleanup: {e}")
